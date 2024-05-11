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

from examples.accelerate_examples.utils.config import get_pretrain_config_file

pretrain_args = PreTrainingArguments.from_yaml(get_pretrain_config_file())

os.makedirs(pretrain_args.save_dir, exist_ok=True)

pretrainer = PreTrainer(pretrain_args=pretrain_args)
pretrainer.train()
