# llama_factory特性支持列表

- 是否支持：llama_factory特性支持列表套件角度出发，昇腾是否完成必要的适配工作，以支持该特性
- 端到端是否支持：从端到端特性使用角度出发，各环节功能是否完全打通。例如以SFT为例，llama_factory侧昇腾是否支持，Pytorch侧昇腾是否支持

## 训练方式

| 一级特性       | 是否支持 |端到端是否支持| 备注     |
|------------|------|--|--------|
| Pre-Training      | 支持   | 支持 |        |
| Supervised Fine-Tuning  | 支持   | 支持 |        |
| Reward Modeling       | 不支持  | 不支持 |  |
| PPO Training       | 不支持   | 不支持 |        |
| DPO Training       | 不支持   | 不支持 |        |
| KTO Training       | 不支持   | 不支持 |        |
| ORPO Training       | 不支持   | 不支持 |        |
| SimPO Training       | 不支持   | 不支持 |        |

## 微调方式

| 一级特性       | 是否支持 |端到端是否支持| 备注     |
|------------|-----|--|--------|
| Full-tuning      | 支持  | 支持 |        |
| LoRA       | 支持  | 支持 |  |
| Freeze-tuning  | 不支持  | 不支持 |        |
| QLoRA       | 不支持  | 不支持 |        |
