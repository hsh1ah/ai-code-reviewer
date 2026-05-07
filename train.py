#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The BigCode & HuggingFace Inc. teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Script to instruction fine-tune causal language models on a Hub dataset

Adapted from huggingface/transformers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""

import logging
import math
import os
import random
import sys
from itertools import chain

import datasets
import torch
import transformers
from config import DataArguments, ModelArguments, TrainingArguments
from datasets import load_dataset
from dialogues import get_dialogue_template, mask_user_labels, prepare_dialogue
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          default_data_collator, set_seed)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from utils import StarChatArgumentParser, hf_login

logger = logging.getLogger(__name__)


def main():
    parser = StarChatArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a YAML file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    # parse command line args and yaml file
    elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
    # parse command line args only
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Login to HuggingFace Hub if needed
    hf_login()

    ###########################
    # Detecting last checkpoint
    ###########################
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    #######################
    # Load pretrained model
    #######################
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        use_cache=not training_args.gradient_checkpointing,
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dialogue template
    dialogue_template = get_dialogue_template(data_args.dialogue_template)

    # Resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.use_flash_attention_2:
        from transformers import LlamaForCausalLM
        if isinstance(model, LlamaForCausalLM):
            from transformers.models.llama.modeling_llama import LlamaFlashAttention2
            model.config._flash_attn_2_enabled = True

    #######################
    # Load and preprocess the dataset
    #######################
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocess datasets
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    # Prepare dialogue function
    def prepare_dialogue_text(examples):
        dialogues = [prepare_dialogue(turns, dialogue_template) for turns in examples["turns"]]
        return {"text": dialogues}

    tokenized_datasets = raw_datasets.map(
        prepare_dialogue_text,
        remove_columns=column_names,
        desc="Preparing dialogues",
    )

    # Tokenize function
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=data_args.max_seq_length,
            truncation=True,
            padding=False,
        )
        # Mask user labels
        masked_labels = mask_user_labels(
            model_inputs["input_ids"],
            tokenizer,
            dialogue_template,
        )
        model_inputs["labels"] = masked_labels
        return model_inputs

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = tokenized_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    #######################
    # Initialize Trainer
    #######################
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    ###############
    # Training loop
    ###############
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #################################
    # Create model card & push to Hub
    #################################
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name
            kwargs["dataset_args"] = "default"

    # Store dialogue template so we can load it at deployment time
    dialogue_template.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.save_model(training_args.output_dir)
        trainer.create_model_card(**kwargs)

    with training_args.main_process_first(desc="Generate a sample from the model"):
        inputs = tokenizer(
            "<|system|>\n<|end|>\n<|user|>\nHow many helicopters can a human eat in one sitting?<|end|>\n<|assistant|>",
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(training_args.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
        )
        logger.info(f"=== SAMPLE OUTPUT ==\n\n{tokenizer.decode(outputs[0], skip_special_tokens=False)}")


if __name__ == "__main__":
    main()
