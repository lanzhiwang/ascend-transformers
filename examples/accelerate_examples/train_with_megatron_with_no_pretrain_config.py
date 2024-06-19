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

DATA_PATH = 'xxx'
BASE_MODEL = 'xxx'
SAVE_PATH = 'xxx'
IS_MEGATRON_DATASET = True
PROJECT_NAME = 'clm_no_trainer'

megatron_train_args = {
    "tp_degree": 4,
    "pp_degree": 1,
    "gradient_clipping": 1.0,
    "num_micro_batches": 8,
    "use_distributed_optimizer": False,
    "sequence_parallelism": True,
    "other_megatron_args": {
        "tokenizer_model": os.path.join(BASE_MODEL, "tokenizer.model"),
        "finetune": False,
        "recompute_granularity": "full",
        "recompute_method": "block",
        "recompute_num_layers": 32,
        "optimizer": "adam",
        "lr": 1e-5,
        "min_lr": 1e-6,
        "adam_beta2": 0.95,
        "add_bias_linear": False,
        "async_tensor_model_parallel_allreduce": True,
        "attention_dropout": 0.0,
        "attention_softmax_in_fp32": True,
        "bias_gelu_fusion": False,
        "ffn_hidden_size": 11008,
        "hidden_dropout": 0.0,
        "init_method_std": 0.01,
        "initial_loss_scale": 65536.0,
        "lr_decay_style": "cosine",
        "lr_warmup_fraction": 0.01,
        "masked_softmax_fusion": False,
        "normalization": "RMSNorm",
        "split": "100,0,0",
        "swiglu": True,
        "untie_embeddings_and_output_weights": True,
        "use_flash_attn": True,
        "weight_decay": 0.1,
        "no_load_optim": True,
        "no_load_rng": True,
    }
}

os.makedirs(SAVE_PATH, exist_ok=True)

pretrain_args = PreTrainingArguments(num_training_steps=1000,
                                     micro_batch_size=4,
                                     dp=1,
                                     gradient_accumulation_steps=8,
                                     seq_length=2048,
                                     megatron_dataset_flag=IS_MEGATRON_DATASET,
                                     data_path=DATA_PATH,
                                     save_dir=SAVE_PATH,
                                     save_interval=10000,
                                     eval_interval=10000,
                                     openmind_model_path=BASE_MODEL,
                                     plugin_args=megatron_train_args,
                                     )

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
