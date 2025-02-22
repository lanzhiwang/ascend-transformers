# coding=utf-8
# Copyright 2023-present Huawei Technologies Co., Ltd
# Copyright 2021-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
import unittest
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401

from accelerate import Accelerator, DistributedDataParallelKwargs, GradScalerKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import AutocastKwargs, KwargsHandler, clear_environment   # TorchDynamoPlugin
from accelerate.test_utils import execute_subprocess_async


@dataclass
class MockClass(KwargsHandler):
    a: int = 0
    b: bool = False
    c: float = 3.0


class KwargsHandlerTestForNPU(unittest.TestCase):
    def test_kwargs_handler(self):
        # If no defaults are changed, `to_kwargs` returns an empty dict.
        self.assertDictEqual(MockClass().to_kwargs(), {})
        self.assertDictEqual(MockClass(a=2).to_kwargs(), {"a": 2})
        self.assertDictEqual(MockClass(a=2, b=True).to_kwargs(), {"a": 2, "b": True})
        self.assertDictEqual(MockClass(a=2, c=2.25).to_kwargs(), {"a": 2, "c": 2.25})

    def test_grad_scaler_kwargs(self):
        scaler_handler = GradScalerKwargs(init_scale=1024, growth_factor=2)
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[scaler_handler])
        print(accelerator.use_fp16)
        scaler = accelerator.scaler

        # Check the kwargs have been applied
        self.assertEqual(scaler._init_scale, 1024.0)
        self.assertEqual(scaler._growth_factor, 2.0)

        # Check the other values are at the default
        self.assertEqual(scaler._backoff_factor, 0.5)
        self.assertEqual(scaler._growth_interval, 2000)
        self.assertEqual(scaler._enabled, True)

    def test_autocast_kwargs(self):
        kwargs = AutocastKwargs(enabled=False)
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="fp16")

        a_float32 = torch.rand((8, 8), device=accelerator.device)
        b_float32 = torch.rand((8, 8), device=accelerator.device)
        c_float32 = torch.rand((8, 8), device=accelerator.device)
        d_float32 = torch.rand((8, 8), device=accelerator.device)

        with accelerator.autocast():
            e_float16 = torch.mm(a_float32, b_float32)
            assert e_float16.dtype == torch.float16

            with accelerator.autocast(autocast_handler=kwargs):
                # Convert e_float16 to float32
                f_float32 = torch.mm(c_float32, e_float16.float())
                assert f_float32.dtype == torch.float32

            g_float16 = torch.mm(d_float32, f_float32)
            # We should be back in fp16
            assert g_float16.dtype == torch.float16

    def test_ddp_kwargs(self):
        cmd = ["torchrun", f"--nproc_per_node={torch.npu.device_count()}", inspect.getfile(self.__class__)]
        execute_subprocess_async(cmd, env=os.environ.copy())

    # Torch Dynamo is not supported by NPU now.
    def test_torch_dynamo_plugin(self):
        pass


if __name__ == "__main__":
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])
    model = torch.nn.Linear(100, 200)
    model = accelerator.prepare(model)

    # Check the values changed in kwargs
    error_msg = ""
    observed_bucket_cap_map = model.bucket_bytes_cap // (1024 * 1024)
    if observed_bucket_cap_map != 15:
        error_msg += f"Kwargs badly passed, should have `15` but found {observed_bucket_cap_map}.\n"
    if model.find_unused_parameters is not True:
        error_msg += f"Kwargs badly passed, should have `True` but found {model.find_unused_parameters}.\n"

    # Check the values of the defaults
    if model.dim != 0:
        error_msg += f"Default value not respected, should have `0` but found {model.dim}.\n"
    if model.broadcast_buffers is not True:
        error_msg += f"Default value not respected, should have `True` but found {model.broadcast_buffers}.\n"
    if model.gradient_as_bucket_view is not False:
        error_msg += f"Default value not respected, should have `False` but found {model.gradient_as_bucket_view}.\n"

    # Raise error at the end to make sure we don't stop at the first failure.
    if len(error_msg) > 0:
        raise ValueError(error_msg)
