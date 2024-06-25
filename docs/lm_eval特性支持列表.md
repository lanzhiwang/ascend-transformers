# lm_eval特性支持列表

- 是否支持：lm_eval套件角度出发，昇腾是否完成必要的适配工作，以支持该特性
- 端到端是否支持：从端到端特性使用角度出发，各环节功能是否完全打通。例如以多级多卡运行评估为例，lm_eval侧昇腾是否支持，Pytorch侧昇腾是否支持


| 一级特性       | 是否支持 |端到端是否支持| 备注                |
|------------|----|--|-------------------|
| 单机单卡       | 支持 | 支持 | 只支持huggingface侧模型 |
| 多级多卡       | 支持 | 支持 | 只支持huggingface侧模型                 |
| 单机多卡       | 支持 | 支持 | 只支持huggingface侧模型                 |

## 脚本启动示例

### 单机单卡

```shell
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen1.5-0.5B \
    --tasks hellaswag \
    --device npu:0 \
    --batch_size 8
```
具体的device可根据实际情况设定，目前原生适配代码支持设定不同卡，例如"npu:1"，"npu:5"，"npu:7"等

### 多机多卡

```shell
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen1.5-0.5B \
    --tasks hellaswag \
    --batch_size 16
```

### 单机多卡

单机多卡基于`device_map="auto"`，分割权重到多卡，从而对推理加速。

```shell
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen1.5-0.5B,parallelize=True \
    --tasks hellaswag \
    --batch_size 16
```

同时由于lm_eval推理脚本源码问题，推理时输入与权重可能不在同一张卡上导致报错，一般可以采取以下方式规避。

- 控制算子执行时启动同步模式

  ```shell
  export ASCEND_LAUNCH_BLOCKING=1
  ```

- 重新执行lm_eval命令，此时log会明确说明哪张卡上权重与输入存在冲突
- 在实际运行命令中添加对应卡号，重新运行lm_eval命令

  ```shell
  lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen1.5-0.5B,parallelize=True \
    --device "npu:7" \
    --tasks hellaswag \
    --batch_size 16
  ```
