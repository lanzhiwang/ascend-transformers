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

import accelerate_npu
from openmind import PreTrainingArguments, PreTrainer
from pretrain_gpt import train_valid_test_datasets_provider, get_batch as megatron_gpt_get_batch, \
    model_provider as megatron_gpt_model_provider, loss_func as megatron_gpt_loss_func
from examples.accelerate_examples.utils.config import get_pretrain_config_file

pretrain_args = PreTrainingArguments.from_yaml(get_pretrain_config_file())
os.makedirs(pretrain_args.save_dir, exist_ok=True)
train_valid_test_datasets_provider.is_distributed = True
# Users can customize functions such as get_batch and loss_function.
# For example, these user-defined functions used here come from the megatron pretrain_gpt file.
# So we should use '--no-use-pep517' to pip install nvidia's megatron from source
pretrain_args.update_distributed_train_args(
    extra_args={
        "custom_megatron_datasets_provider_function": train_valid_test_datasets_provider,
        "custom_get_batch_function": megatron_gpt_get_batch,
        "custom_model_provider_function": megatron_gpt_model_provider,
        "custom_loss_function": megatron_gpt_loss_func,
    }
)

pretrainer = PreTrainer(pretrain_args=pretrain_args)
pretrainer.train()
