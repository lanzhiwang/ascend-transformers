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

from openmind import AutoTokenizer


def get_tokenizer(tokenizer_path, pad_token_id=0, use_fast=True, logger=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast)
    tokenizer.pad_token_id = pad_token_id  # 0, unk. we want this to be different from the eos token
    if logger:
        logger.info(f"tokenizer.pad_token_id:{tokenizer.pad_token_id} tokenizer.pad_token:{tokenizer.pad_token}")
        logger.info(f"tokenizer.bos_token_id:{tokenizer.bos_token_id} tokenizer.bos_token:{tokenizer.bos_token}")
        logger.info(f"tokenizer.eos_token_id:{tokenizer.eos_token_id} tokenizer.eos_token:{tokenizer.eos_token}")
    return tokenizer
