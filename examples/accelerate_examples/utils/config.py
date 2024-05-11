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

import argparse

from examples.accelerate_examples.utils.tools import print_in_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a openmind model on a causal language modeling task")
    parser.add_argument(
        "--pretrain_config_file",
        type=str,
        default=None,
        help="The pretrain config file path.",
    )
    args = parser.parse_args()
    return args


def get_pretrain_config_file():
    args = parse_args()
    if args.pretrain_config_file:
        print_in_main_process("*" * 15 + f"The pretrain config file path is : {args.pretrain_config_file}" + "*" * 15)
        return args.pretrain_config_file
    print_in_main_process("*" * 15 + "Warning: Pretrain config file is not configured." + "*" * 15)
    return None
