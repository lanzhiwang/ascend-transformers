# Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# openMind Accelerate is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os

import torch
from transformers import get_scheduler

import accelerate_npu
from openmind import PreTrainingArguments, PreTrainer

from examples.accelerate_examples.utils.config import get_pretrain_config_file
from examples.accelerate_examples.utils.accelerator import make_accelerator
from examples.accelerate_examples.utils.data import make_train_and_eval_dataloader
from examples.accelerate_examples.utils.model import read_openmind_model
from examples.accelerate_examples.utils.tokenizer import get_tokenizer

pretrain_args = PreTrainingArguments.from_yaml(get_pretrain_config_file())

os.makedirs(pretrain_args.save_dir, exist_ok=True)

accelerator = make_accelerator(pretrain_args=pretrain_args)

tokenizer = get_tokenizer(tokenizer_path=pretrain_args.openmind_model_path, use_fast=False)
transformer_dataloader_config = pretrain_args.get_dataloader_config()
train_dataloader, eval_dataloader = make_train_and_eval_dataloader(
    dataloader_config=transformer_dataloader_config,
    micro_batch_size=pretrain_args.micro_batch_size * pretrain_args.dp,
    data_files=pretrain_args.data_path,
    max_length=pretrain_args.seq_length,
    tokenizer=tokenizer,
    accelerator=accelerator
)

model = read_openmind_model(model_path=pretrain_args.openmind_model_path,
                            torch_dtype=pretrain_args.get_torch_dtype(),
                            accelerator=accelerator)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=1000,
)

pretrainer = PreTrainer(pretrain_args=pretrain_args,
                        train_dataloader=train_dataloader,
                        eval_dataloader=eval_dataloader,
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        )
pretrainer.train()
