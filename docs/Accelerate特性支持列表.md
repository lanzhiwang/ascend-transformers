# Accelerate特性支持列表

- 是否支持：accelerate套件角度出发，昇腾是否完成必要的适配工作，以支持该特性
- 端到端是否支持：从端到端特性使用角度出发，各环节功能是否完全打通。例如以DeepSpeed Zero2为例，accelerate侧昇腾是否支持，DeepSpeed侧昇腾是否支持，Pytorch侧昇腾是否支持


## AMP


| 二级特性        | 是否支持 |端到端是否支持| 备注                       |
|-------------|----|--|--------------------------|
| no(deafult) | 支持 | 支持 |  |
| fp16        | 支持 | 支持 |  |
| bf16        | 支持 | 支持 |  |
| apex        | 支持 | 支持 |  |


## DDP

| 一级特性 | 是否支持 | 端到端是否支持 | 备注                                                                                                                                |
|------|------|---------|-----------------------------------------------------------------------------------------------------------------------------------|
| DDP  | 支持   | 支持      | |



## DeepSpeed

注：accelerate适配已完成，功能由deepspeed提供

| 二级特性           | 是否支持 | 端到端是否支持 | 备注 |
|----------------|-----|---------|----|
| Zero1          | 支持  | 支持      |    |
| Zero2          | 支持  | 支持      |    |
| Zero3          | 支持  | 不支持     |    |
| offload_to_nvme | 支持  | 不支持     |    |
| offload_to_cpu | 支持  | 支持      |    |


## FSDP

| 二级特性                               | 是否支持 |端到端是否支持| 备注 |
|------------------------------------|-----|--|----|
| FULL_SHARD                         | 支持  |支持 |    |
| SHARD_GRAD_OP                      | 支持  | 支持 |    |
| NO_SHARD                           | 支持  |  支持|    |
| HYBRID_SHARD                       | 支持  | 支持 |    |
| HYBRID_SHARD_ZERO2                 | 支持  | 支持 |    |
| CPU offload                        | 支持  | 支持 |    |
| fsdp_auto_wrap_policy              | 支持  | 支持 | transformer_based_wrap, size_based_wrap, no_wrap |
| fsdp_transformer_layer_cls_to_wrap | 支持  | 支持 | 跟transformer_based_wrap配合使用 |
| fsdp_min_num_params                | 支持  |支持 |跟size_based_wrap配合使用   |
| fsdp_backward_prefetch_policy      | 支持  |支持 |  backward_pre, backward_post, no_prefetch |
| fsdp_forward_prefetch              | 支持  |支持| 静态图下使用             |
| fsdp_state_dict_type               | 支持  |支持 |  full_state_dict, local_state_dict, shared_state_dict |
| fsdp_use_orig_params               | 支持  |支持 |    |
| fsdp_cpu_ram_efficient_loading     | 支持  |支持 | 仅用于transformers模型   |




## bnb

| 二级特性              | 是否支持 | 端到端是否支持 | 备注 |
|-------------------|------|---------|----|
| bnb.nn.Linear8bit | 不支持  | 不支持     |    |
| bnb.nn.Linear4bit | 不支持  | 不支持     |    |

## Local Sgd

| 二级特性     | 是否支持 | 端到端是否支持 | 备注 |
|----------|-----|--------|----|
| LocalSGD | 支持  | 支持     |    |



## Launchers

| 二级特性             | 是否支持 | 端到端是否支持 | 备注                                |
|------------------|------|--------|-----------------------------------|
| test_lauch       | 不支持  | 不支持     |                                   |
| notebook_launcher | 不支持   | 不支持     | 该特性用于在colab或kaggle环境下使用accelerate |
| debug_launcher   | 不支持   | 不支持     |                                   |


## DataLoader

| 二级特性                   | 是否支持 | 端到端是否支持 | 备注       |
|------------------------|-----|--------|----------|
| SeedableRandomSampler  | 支持  | 支持     |          |
| BatchSamplerShard      | 支持  | 支持     |          |
| IterableDatasetShard   | 支持  | 支持     |          |
| DataLoaderStateMixin   | 支持  | 支持     |          |
| DataLoaderSharder      | 支持  | 支持     |          |
| MpDdeviceLoaderWrapper | 不支持  | 不支持     | 该特性用于TPU |
| DataLoaderDispatcher   | 支持  | 支持     |          |


## Logging

| 二级特性                 | 是否支持 | 端到端是否支持 | 备注 |
|----------------------|-----|--------|----|
| MultiProcessAdapter  | 支持  | 支持     |    |

## Optimizer

| 二级特性                 | 是否支持 | 端到端是否支持 | 备注 |
|----------------------|-----|--------|----|
| AcceleratedOptimizer | 支持  | 支持     |    |


## Scheduler 

| 二级特性                 | 是否支持 | 端到端是否支持 | 备注 |
|----------------------|-----|--------|----|
| AcceleratedScheduler | 支持  | 支持     |    |



## Tracking

| 二级特性               | 是否支持 | 端到端是否支持 | 备注 |
|--------------------|-----|--------|----|
| GeneralTracker     | 支持  | 支持     |    |
| TensorboardTracker | 支持  | 支持     |    |
| WandBTracker       | 支持  | 支持     |    |
| CometMLTracker     | 支持  | 支持     |    |
| AimTracker         | 支持  | 支持     |    |
| MLflowTracker      | 支持  | 支持     |    |
| ClearMLTracker     | 支持  | 支持     |    |
| DVCLiveTracker     | 支持  | 支持     |    |

## Checkpointing

| 二级特性                   | 是否支持 | 端到端是否支持 | 备注 |
|------------------------|-----|--------|----|
| save_accelerator_state | 支持  | 支持     |    |
| load_accelerator_state | 支持  | 支持     |    |
| save_custom_state      | 支持  | 支持     |    |
| load_custom_state      | 支持  | 支持     |    |

## Hook

*hook和pipeline parallelisom联合使用，依赖单进程多卡特性，如果有问题可到[Ascend/pytorch](https://gitee.com/ascend/pytorch/)下提issue解决*

| 二级特性                               | 是否支持 | 端到端是否支持 | 备注 |
|------------------------------------|-----|---------|----|
| SequentialHook                     | 支持  | 不支持     |    |
| add_hook_to_module                 | 支持  | 不支持     |    |
| remove_hook_from_module            | 支持  | 不支持     |    |
| AlignDevicesHook                   | 支持  | 不支持     |    |
| attach_execution_device_hook       | 支持  | 不支持     |    |
| attach_align_device_hook           | 支持  | 不支持     |    |
| remove_hook_from_submodules        | 支持  | 不支持     |    |
| attach_align_device_hook_on_blocks | 支持  | 不支持     |    |
| CpuOffload                         | 支持  | 不支持     |    |
| UserCpuOffloadHook                 | 支持  | 不支持     |    |


## Big Modeling 

*依赖单进程多卡特性*

| 二级特性                 | 是否支持 | 端到端是否支持 | 备注 |
|----------------------|-----|--------|----|
| init_empty_weights   | 支持  | 不支持    |    |
| init_on_device       | 支持  | 不支持    |    |
| cpu_offload          | 支持  | 不支持    |    |
| cpu_offload_with_hook | 支持  | 不支持    |    |
| disk_offload         | 支持  | 不支持    |    |
| dispatch_model       | 支持  | 不支持    |    |
| load_checkpoint      | 支持  | 不支持    |    |

## CLI

*CLI指代accelerate支持的command line*

| 二级特性                       | 是否支持 | 端到端是否支持 | 备注 |
|----------------------------|-----|---|----|
| accelerate config          | 支持  | 支持 |    |
| accelerate config update   | 支持  | 支持 |    |
| accelerate env             | 支持  | 支持 |    |
| accelerate launch          | 支持  | 支持 |    |
| accelerate estimate-memory | 支持  | 支持 |    |
| accelerate tpu-config      | 不支持 |不支持| 适配TPU  |
| accelerate test            | 支持  | 支持 |    |


## Dynamo

不支持

## Megatron-LM

| 二级特性                            | 是否支持 | 端到端是否支持 | 备注 |
|---------------------------------|-----|---------|----|
| is_megatron_lm_available        | 不支持  | 不支持     |    |
| megatorn_lm_plugin              | 不支持  | 不支持     |    |
| megatron_lm_initialize          | 不支持  | 不支持     |    |
| megatron_lm_prepare_data_loader | 不支持  | 不支持     |    |
| megatron_lm_prepare_model       | 不支持  | 不支持     |    |
| megatron_lm_prepare_optimzer    | 不支持  | 不支持     |    |
| megatron_lm_prepare_scheduler   | 不支持  | 不支持     |    |
| MegatronEngine                  | 不支持  | 不支持     |    |
| model.forward                   | 不支持  | 不支持     |    |
| accelerator.backward            | 不支持  | 不支持     |    |
| optimizer.step                  | 不支持  | 不支持     |    |
| scheduler.step                  | 不支持  | 不支持     |    |
| accelerator.load_state          | 不支持  | 不支持     |    |
| accelerator.save_state          | 不支持  | 不支持     |    |