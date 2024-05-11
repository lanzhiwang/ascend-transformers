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
SCRIPT_PATH="${SCRIPT_DIR}"/train_with_megatron.py
ACCELERATE_CONF_PATH="$SCRIPT_DIR"/accelerate_config/accelerate_megatron_config.yaml
PRETRAIN_CONF_PATH="$SCRIPT_DIR"/llama2_config/llama2-megatron.yaml

ARGS=`getopt -o s:a:p: --long script_path:,accelerate_conf_path:,pretrain_conf_path: -n "$0" -- "$@"`
eval set -- "${ARGS}"
while true
do
    case "$1" in
        -s|--script_path)
            echo "Option script_path, argument $2";
            SCRIPT_PATH=$2
            shift 2
            ;;
        -a|--accelerate_conf_path)
            echo "Option accelerate_conf_path, argument $2";
            ACCELERATE_CONF_PATH=$2
            shift 2
            ;;
        -p|--pretrain_conf_path)
            echo "Option pretrain_conf_path, argument $2";
            PRETRAIN_CONF_PATH=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

accelerate launch --config_file "${ACCELERATE_CONF_PATH}" "${SCRIPT_PATH}" --pretrain_config_file "${PRETRAIN_CONF_PATH}" | tee "${SCRIPT_DIR}"/train.log