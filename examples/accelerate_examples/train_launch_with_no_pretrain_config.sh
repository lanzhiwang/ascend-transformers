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
#!/bin/bash
export CUDA_VISIBLE_DEVICES_=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_}
export LD_LIBRARY_PATH=/usr/local/lib:/home/anaconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
# Using async gradient all reduce requires setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /xxx/set_env.sh

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
SCRIPT_PATH="${SCRIPT_DIR}"/train_with_megatron_with_no_pretrain_config.py
CONF_PATH="${SCRIPT_DIR}"/accelerate_config/accelerate_megatron_config.yaml

accelerate launch --config_file "${CONF_PATH}" "${SCRIPT_PATH}"
