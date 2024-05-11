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

from accelerate import Accelerator

from examples.accelerate_examples.utils.tools import print_in_main_process


def _get_megatron_lm_plugin(pretrain_args):
    from accelerate.utils import MegatronLMPlugin

    plugin_args = {
        "train_iters": pretrain_args.num_training_steps,
        "seq_length": pretrain_args.seq_length,
        "num_micro_batches": pretrain_args.gradient_accumulation_steps,
        "megatron_dataset_flag": pretrain_args.megatron_dataset_flag,
    }
    if pretrain_args.plugin_args:
        for key, value in pretrain_args.plugin_args.items():
            if key in plugin_args.keys():
                print_in_main_process(
                    f"WARNING: Make accelerator megatron_lm plugin overriding arguments for "
                    f"{key}:{plugin_args[key]} with {key}:{value}"
                )
            plugin_args[key] = value

    return MegatronLMPlugin(**plugin_args)


def make_accelerator(pretrain_args):
    accelerate_kwargs = {
        "log_with": pretrain_args.report_to,
        "project_dir": pretrain_args.save_dir,
        "mixed_precision": pretrain_args.get_mixed_precision(),
    }
    if os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false") == "true":
        megatron_lm_plugin = _get_megatron_lm_plugin(
            pretrain_args,
        )
        accelerate_kwargs["megatron_lm_plugin"] = megatron_lm_plugin
    else:
        accelerate_kwargs["gradient_accumulation_steps"] = pretrain_args.gradient_accumulation_steps
    return Accelerator(**accelerate_kwargs)
