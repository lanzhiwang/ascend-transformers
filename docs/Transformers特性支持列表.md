# Transformers特性支持列表

- 是否支持：Transformers套件角度出发，昇腾是否完成必要的适配工作，以支持该特性
- 端到端是否支持：从端到端特性使用角度出发，各环节功能是否完全打通。例如以DeepSpeed Zero2为例，Transformers侧昇腾是否支持，DeepSpeed侧昇腾是否支持，Pytorch侧昇腾是否支持


## AMP


| 二级特性        | 是否支持 |端到端是否支持| 备注                              |
|-------------|----|--|---------------------------------|
| no(deafult) | 支持 | 支持 |                                 |
| fp16        | 支持 | 支持 |                                 |
| bf16        | 支持 | 支持 |                                 |
| apex        | 支持 | 支持 | 特性支持受限，具体支持特性请查看[Ascend/apex](https://gitee.com/ascend/apex)|

## DDP 

| 一级特性 | 是否支持 | 端到端是否支持 | 备注                                                                                                                                |
|------|------|---------|-----------------------------------------------------------------------------------------------------------------------------------|
| DDP  | 支持   | 支持      | |


## DP

| 一级特性 | 是否支持 | 端到端是否支持 | 备注  |
|------|------|---------|-----------------|
| DP   | 不支持  | 不支持     |  |

## Peft

| 二级特性            | 是否支持 |端到端是否支持| 备注 |
|-----------------|-----|--|----|
| Lora            | 支持  | 支持 |    |
| IA3             | 支持  | 支持 |    |
| AdaLoRA         | 支持  | 支持 |    |


## Deepspeed

*功能由deepspeed提供, 具体问题可到[Ascend/DeepSpeed](https://gitee.com/ascend/DeepSpeed)提issue解决*

| 二级特性           | 是否支持 | 端到端是否支持 | 备注 |
|----------------|-----|---------|----|
| Zero1          | 支持  | 支持      |    |
| Zero2          | 支持  | 支持      |    |
| Zero3          | 支持  | 不支持     |    |
| offload_to_nvme | 支持  | 不支持     |    |
| offload_to_cpu | 支持  | 支持      |    |



## FSDP

| 二级特性                                                        | 是否支持 |端到端是否支持| 备注 |
|-------------------------------------------------------------|-----|--|----|
| FULL_SHARD                                                  | 支持  | 支持 |    |
| SHARD_GRAD_OP                                               | 支持  | 支持 |    |
| NO_SHARD                                                    | 支持  |支持  |    |
| HYBRID_SHARD                                                | 支持  | 支持 |    |
| HYBRID_SHARD_ZERO2                                          | 支持  | 支持 |    |
| CPU offload                                                 | 支持  | 支持 |    |
| Wrapping policy:<br/>TRANSFORMERS_BASED_WRAP_SIZE_BASD_WRAP | 支持  | 支持 |    |


## TrainingAruguments

| 二级特性               | 是否支持 |端到端是否支持| 备注 |
|----------------|------|--|----|
|output_dir | 支持   | 支持 |    |
|overwrite_output_dir | 支持   | 支持 |    |
|do_train | 支持   | 支持 |    |
|do_eval | 支持   | 支持 |    |
|do_predict | 支持   | 支持 |    |
|evaluation_strategy | 支持   | 支持 |    |
|prediction_loss_only | 支持   | 支持 |    |
|per_device_train_batch_size | 支持   |支持  |    |
|per_device_eval_batch_size | 支持   |支持  |    |
|per_gpu_train_batch_size | 支持   | 支持 |    |
|per_gpu_eval_batch_size | 支持   | 支持 |    |
|gradient_accumulation_steps | 支持   | 支持 |    |
|eval_accumulation_steps | 支持   | 支持 |    |
|eval_delay | 支持   | 支持 |    |
|learning_rate | 支持   | 支持 |    |
|weight_decay | 支持   |支持  |    |
|adam_beta1 | 支持   |支持  |    |
|adam_beta2 | 支持   | 支持 |    |
|adam_epsilon | 支持   |支持  |    |
|max_grad_norm | 支持   |支持  |    |
|num_train_epochs | 支持   |支持  |    |
|max_steps | 支持   | 支持 |    |
|lr_scheduler_type | 支持   |支持  |    |
|lr_scheduler_kwargs | 支持   |支持  |    |
|warmup_ratio | 支持   | 支持 |    |
|warmup_steps | 支持   | 支持 |    |
|log_level | 支持   |支持  |    |
|log_level_replica | 支持   |支持  |    |
|log_on_each_node | 支持   |支持  |    |
|logging_dir | 支持   | 支持 |    |
|logging_strategy | 支持   |支持  |    |
|logging_first_step | 支持   | 支持 |    |
|logging_steps | 支持   |支持  |    |
|logging_nan_inf_filter | 支持   | 支持 |    |
|save_strategy | 支持   |支持  |    |
|save_steps | 支持   | 支持 |    |
|save_total_limit | 支持   |支持  |    |
|save_safetensors | 支持   |支持  |    |
|save_on_each_node | 支持   |支持  |    |
|save_only_model | 支持   |支持  |    |
|no_cuda | 支持   |支持  |    |
|use_cpu | 支持   | 支持 |    |
|use_mps_device | 不支持  | 不支持 |    |
|seed | 支持   | 支持 |    |
|data_seed | 支持   |支持  |    |
|jit_mode_eval | 支持   |支持  |    |
|use_ipex | 不支持  |不支持  |    |
|bf16 | 支持   |支持  |    |
|fp16 | 支持   | 支持 |    |
|fp16_opt_level | 支持   | 支持 |    |
|half_precision_backend | 支持   | 支持 |    |
|bf16_full_eval | 支持   |支持  |    |
|fp16_full_eval | 支持   |支持  |    |
|tf32 | 不支持  | 不支持 |    |
|local_rank | 支持   |支持  |    |
|ddp_backend | 支持   | 支持 |    |
|tpu_num_cores | 不支持  |不支持  |    |
|tpu_metrics_debug | 不支持  |不支持  |    |
|debug | 支持   |支持  |    |
|dataloader_drop_last | 支持   |支持  |    |
|eval_steps | 支持   | 支持 |    |
|dataloader_num_workers | 支持   | 支持 |    |
|dataloader_prefetch_factor | 支持   |支持  |    |
|past_index | 支持   | 支持 |    |
|run_name | 支持   |支持  |    |
|disable_tqdm | 支持   | 支持 |    |
|remove_unused_columns | 支持   |支持  |    |
|label_names | 支持   | 支持 |    |
|load_best_model_at_end | 支持   |支持  |    |
|metric_for_best_model | 支持   | 支持 |    |
|greater_is_better | 支持   |支持  |    |
|ignore_data_skip | 支持   |支持  |    |
|fsdp | 支持   |支持  |    |
|fsdp_min_num_params | 支持   |支持  |    |
|fsdp_config | 支持   | 支持 |    |
|fsdp_transformer_layer_cls_to_wrap | 支持   |支持  |    |
|accelerator_config | 支持   |支持  |    |
|deepspeed | 支持   |支持  |    |
|label_smoothing_factor | 支持   | 支持 |    |
|default_optim | 支持   | 支持 |    |
|optim | 支持   | 支持 |    |
|optim_args | 支持   | 支持 |    |
|adafactor | 支持   |支持  |    |
|group_by_length | 支持   |支持  |    |
|length_column_name | 支持   | 支持 |    |
|report_to | 支持   | 支持 |    |
|ddp_find_unused_parameters | 支持   |支持  |    |
|ddp_bucket_cap_mb | 支持   | 支持 |    |
|ddp_broadcast_buffers | 支持   |支持  |    |
|dataloader_pin_memory | 支持   |支持  |    |
|dataloader_persistent_workers | 支持   | 支持 |    |
|skip_memory_metrics | 支持   | 支持 |    |
|use_legacy_prediction_loop | 支持   | 支持 |    |
|push_to_hub | 支持   | 支持 |    |
|resume_from_checkpoint | 支持   |支持  |    |
|hub_model_id | 支持   | 支持 |    |
|hub_strategy | 支持   |支持  |    |
|hub_token | 支持   | 支持 |    |
|hub_private_repo | 支持   | 支持 |    |
|hub_always_push | 支持   |支持  |    |
|gradient_checkpointing | 支持   | 支持 |    |
|gradient_checkpointing_kwargs | 支持   |支持  |    |
|include_inputs_for_metrics | 支持   |支持  |    |
|fp16_backend | 支持   | 支持 |    |
|push_to_hub_model_id | 支持   | 支持 |    |
|push_to_hub_organization | 支持   | 支持 |    |
|push_to_hub_token | 支持   |支持  |    |
|_n_gpu | 支持   |  支持|    |
|mp_parameters | 不支持  |不支持  |    |
|auto_find_batch_size | 支持   | 支持 |    |
|full_determinism | 不支持  |不支持  |    |
|torchdynamo | 不支持  |不支持  |    |
|ray_scope | 不支持  |不支持  |    |
|ddp_timeout | 支持   | 支持 |    |
|torch_compile | 不支持  | 不支持 |    |
|torch_compile_backend | 不支持  |  不支持|    |
|torch_compile_mode | 不支持  | 不支持 |    |
|dispatch_batches | 支持   |支持  |    |
|split_batches | 支持   |支持 |    |
|include_tokens_per_second | 支持   |支持  |    |
|include_num_input_tokens_seen | 支持   | 支持 |    |
|neftune_noise_alpha | 不支持  |不支持  |    |
