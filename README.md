# Hugging Face
## 简介
:tada: Hugging Face 核心套件  [transformers](https://github.com/huggingface/transformers) 、  [accelerate](https://github.com/huggingface/accelerate) 、 [peft](https://github.com/huggingface/peft) 、  [trl](https://github.com/huggingface/trl)  已原生支持 Ascend NPU，相关功能陆续补齐中。

本仓用于发布最新进展、需求/问题跟踪、测试用例。


## 更新日志

[23/09/13] :fire:  accelerate 支持 Ascend NPU 使用 BF16 格式进行混合精度训练

[23/09/05] :fire:  [transformers](https://github.com/huggingface/transformers) 支持 Ascend NPU 使用 apex 进行混合精度训练

[23/08/23] :tada: 现在 `transformers>=4.32.0` `accelearte>=0.22.0`、`peft>=0.5.0`、`trl>=0.5.0`原生支持 Ascend NPU！通过 `pip install` 即可安装体验

[23/08/11] :sparkles:  [trl](https://github.com/huggingface/trl) 原生支持 Ascend NPU，请源码安装体验

[23/08/05] :sparkles: [accelerate](https://github.com/huggingface/accelerate) 支持 Ascend NPU 使用 FSDP(pt-2.1, Experimental)

[23/08/02] :sparkles:  [peft](https://github.com/huggingface/peft) 原生支持 Ascend NPU 的加载 adapter，请源码安装体验

[23/07/19] :sparkles:  [transformers](https://github.com/huggingface/transformers) 原生支持 Ascend NPU 的单卡/多卡/amp 训练，请源码安装体验

[23/07/12] :sparkles:  [accelerate](https://github.com/huggingface/accelerate) 原生支持 Ascend NPU 的单卡/多卡/amp 训练，请源码安装体验



## Transformers

### 使用说明

当前  [transformers](https://github.com/huggingface/transformers)  训练流程已原生支持 Ascend NPU，这里以参考示例中 [text-classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) 任务为例说明如何在 Ascend NPU 微调 bert 模型。

#### 环境准备

1. 请参考《[Pytorch框架训练环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes)》准备环境，要求 python>=3.8, PyTorch >= 1.9

2. 安装 transformers

   ```
   pip3 install -U transformers
   ```

#### 单卡训练

获取[text-classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)训练脚本并安装相关依赖

```
git clone https://github.com/huggingface/transformers.git
cd examples/pytorch/text-classification
pip install -r requirements.txt
```

执行单卡训练

参考 [text-classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) README，执行：

```bash
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

#### 多卡训练

执行多卡训练

```bash
export TASK_NAME=mrpc

python -m torch.distributed.launch --nproc_per_node=8 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

#### 混合精度训练

如果希望使用混合精度，请传入训练参数 `--fp16` 

```bash
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --fp16 \
  --output_dir /tmp/$TASK_NAME/
```

如果希望使用apex进行混合精度训练，请先确保安装支持 Ascend NPU 的 apex 包，然后传入 `--fp16` 和 `--half_precision_backend apex`

```bash
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --fp16 \
  --half_precision_backend apex \
  --output_dir /tmp/$TASK_NAME/
```

### 支持的特性

- [x] single NPU

- [x] multi-NPU on one node (machine)

- [x] FP16 mixed precision

- [x] PyTorch Fully Sharded Data Parallel (FSDP) support (Partially support,Experimental)

  > need more test

- [ ] DeepSpeed support (Experimental)
- [ ] Big model inference

更多支持的特性请查看[Transformers特性支持列表](docs/Transformers特性支持列表.md)，该列表特性在`transformers==4.37.1`下完成特性测试。

## Accelerate
Accelerate 已经原生支持 Ascend NPU，这里给出基本的使用方法，请参考 [accelerate-usage-guides](https://huggingface.co/docs/accelerate/main/en/usage_guides/explore) 的解锁更多用法。

### 使用说明

执行 `accelerate env` 查看当前环境，确保 `PyTorch NPU available` 为 `True`

```shell
$ accelerate env
-----------------------------------------------------------------------------------------------------------------------------------------------------------

Copy-and-paste the text below in your GitHub issue

- `Accelerate` version: 0.23.0.dev0
- Platform: Linux-5.10.0-60-18.0.50.oe2203.aarch64-with-glibc2.26
- Python version: 3.8.17
- Numpy version: 1.24.4
- PyTorch version (GPU?): 2.1.0.dev20230817 (False)
- PyTorch XPU available: False
- PyTorch NPU available: True
- System RAM: 2010.33 GB
- `Accelerate` default config:
          Not found
```

**场景一：** 单卡训练

在您的设备上执行 `accelerate config`
```shell
$ accelerate config
-----------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]:NO
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all
Do you wish to use FP16 or BF16 (mixed precision)?
fp16
```
设置成功后将会得到如下提示信息：
```text
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
```
这将生成一个配置文件，在执行操作时将自动使用该文件来设置训练选项
```bash
accelerate launch my_script.py --args_to_my_script
```
举个例子，以下是在NPU上运行NLP示例`examples/nlp_example.py`(在accelerate根目录下)。在执行完`accelerate config`后生成的`default_config.yaml`如下：
```text
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'fp16'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
```bash
accelerate launch examples/nlp_example.py
```

**场景二：** 单机多卡训练
同上，首先执行`accelerate config`

```text
$ accelerate config
-----------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-NPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: NO
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want use FullyShardedDataParallel? [yes/NO]:NO
How many NPU(s) should be used for distributed training? [1]:4
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all
Do you wish to use FP16 or BF16 (mixed precision)?
fp16
```
生成的配置如下
```text
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_NPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'fp16'
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
运行NLP示例`examples/nlp_example.py`(在accelerate根目录下)。
```bash
accelerate launch examples/nlp_example.py
```

### 支持的特性

- [x] single NPU

- [x] multi-NPU on one node (machine)

- [x] FP16/BF16 mixed precision

- [x] PyTorch Fully Sharded Data Parallel (FSDP) support (Partially support,Experimental)

  > need more test

- [ ] DeepSpeed support (Experimental)
- [ ] Big model inference
- [ ] Quantization

更多支持的特性请查看[Accelerate特性支持列表](docs/Accelerate特性支持列表.md)，该列表特性在`accelerate==0.26.1`下完成特性测试。

## TRL

TRL支持的特性请查看[Trl特性支持列表](docs/Trl特性支持列表.md)， 该列表特性在`trl==0.7.11.dev0(源码安装)`下完成特性测试。

## 安全声明

### 运行用户建议
出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用。

### 文件权限控制
1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控，管控权限可参考表1进行设置。

    表1 文件（夹）各场景权限管控推荐最大值

    | 类型           | linux权限参考最大值 |
    | -------------- | ---------------  |
    | 用户主目录                        |   750（rwxr-x---）            |
    | 程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）             |
    | 程序文件目录                      |   550（r-xr-x---）            |
    | 配置文件                          |  640（rw-r-----）             |
    | 配置文件目录                      |   750（rwxr-x---）            |
    | 日志文件(记录完毕或者已经归档)        |  440（r--r-----）             | 
    | 日志文件(正在记录)                |    640（rw-r-----）           |
    | 日志文件目录                      |   750（rwxr-x---）            |
    | Debug文件                         |  640（rw-r-----）         |
    | Debug文件目录                     |   750（rwxr-x---）  |
    | 临时文件目录                      |   750（rwxr-x---）   |
    | 维护升级文件目录                  |   770（rwxrwx---）    |
    | 业务数据文件                      |   640（rw-r-----）    |
    | 业务数据文件目录                  |   750（rwxr-x---）      |
    | 密钥组件、私钥、证书、密文文件目录    |  700（rwx—----）      |
    | 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
    | 加解密接口、加解密脚本            |   500（r-x------）        |

### 运行安全声明

本章节对部件实际使用过程中涉及的风险进行声明。

1. 建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。


### 公网地址说明

  代码涉及公网地址参考 [public_address_statement.md](https://gitee.com/ascend/transformers/blob/master/public_address_statement.md)


## FQA

- 使用`transformers`、`accelerate`、`trl`等套件时仅需在您的脚本入口处添加 `import torch, torch_npu` 请勿使用 `from torch_npu.contrib import transfer_to_npu`
  ```python
   import torch
   ipmort torch_npu
   # original code, no from torch_npu.contrib import transfer_to_npu
  ```
- 使用混合精度训练时建议开启非饱和模式：`export INF_NAN_MODE_ENABLE=1`