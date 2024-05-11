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

from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers


def alpaca_prompt(data_point):
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{data_point['instruction']}"
    )

    input_str = data_point.get("input", "")
    if input_str:
        prompt += f"""\n\n### Input:\n{data_point['input']}"""

    prompt += """\n\n### Response:\n"""

    output = data_point.get("output", "")

    return prompt, output


def _prompt_to_token(tokenizer, prompt, response, max_length=512, add_eos_token=True):
    prompt_result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    response_result = tokenizer(
        response,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    result = {
        "input_ids": prompt_result["input_ids"] + response_result["input_ids"][1:],
        "attention_mask": prompt_result["attention_mask"] + response_result["attention_mask"][1:],
    }
    result["input_ids"] = result["input_ids"][:max_length]
    result["attention_mask"] = result["attention_mask"][:max_length]

    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < max_length and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    len_prompt = len(prompt_result["input_ids"])
    result["labels"][:len_prompt] = [-100] * len_prompt  # -100 will be automatically ignored by PyTorch loss functions
    result["labels"] = result["labels"][:max_length]

    return result


def get_generate_and_tokenize_prompt_fn(tokenizer, max_length=512):
    def generate_and_tokenize_prompt(data_point):
        prompt, response = alpaca_prompt(data_point)
        tokenized_full_prompt = _prompt_to_token(
            tokenizer=tokenizer, prompt=prompt, response=response, max_length=max_length, add_eos_token=True
        )
        return tokenized_full_prompt

    return generate_and_tokenize_prompt


def make_train_and_eval_dataloader(
    dataloader_config,
    micro_batch_size,
    data_files,
    max_length,
    test_size=200,
    accelerator=None,
    logger=None,
    tokenizer=None,
    is_megatron_dataset=False,
):
    if is_megatron_dataset:
        from accelerate.utils import MegatronLMDummyDataLoader

        megatron_dataloader = MegatronLMDummyDataLoader(**dataloader_config)
        train_dataloader = megatron_dataloader
        eval_dataloader = megatron_dataloader
        if accelerator:
            accelerator.state.megatron_lm_plugin.megatron_dataset_flag = True
    else:
        if accelerator:
            if logger is None:
                from accelerate.logging import get_logger

                logger = get_logger(__name__)
            with accelerator.main_process_first():
                train_dataset, eval_dataset = make_datasets(tokenizer, data_files, max_length, test_size, logger)
        else:
            train_dataset, eval_dataset = make_datasets(tokenizer, data_files, max_length, test_size, logger)

        data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, **dataloader_config)
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=micro_batch_size, num_workers=2
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=micro_batch_size, num_workers=2)
    return train_dataloader, eval_dataloader


def make_datasets(tokenizer, data_files, max_length, test_size=200, logger=None):
    if logger:
        logger.info("Start handle dataset.", main_process_only=False)
    data = load_dataset("json", data_files=data_files)
    generate_and_tokenize_prompt = get_generate_and_tokenize_prompt_fn(tokenizer=tokenizer, max_length=max_length)
    data = data["train"].train_test_split(test_size=test_size, shuffle=True, seed=42)
    data["train"] = data["train"].map(
        generate_and_tokenize_prompt, remove_columns=data["train"].column_names, num_proc=2
    )
    data["test"] = data["test"].map(generate_and_tokenize_prompt, remove_columns=data["test"].column_names, num_proc=2)
    if logger:
        logger.info(data)
    return data["train"], data["test"]
