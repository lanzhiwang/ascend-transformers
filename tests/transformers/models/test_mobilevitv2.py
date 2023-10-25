# coding=utf-8
# Copyright 2023-present Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch MobileVitV2 model. """

import unittest

from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import is_torch_available, is_vision_available, cached_property

if is_torch_available():
    import torch
    from transformers import MobileViTV2ForImageClassification

if is_vision_available():
    from PIL import Image
    from transformers import MobileViTImageProcessor

MODEL_NAME_OR_PATH = "apple/mobilevitv2-1.0-imagenet1k-256"


def prepare_img():
    image = Image.open("000000039769.png")
    return image


@require_torch
@require_vision
class MobileViTV2ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            MobileViTImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
            if is_vision_available()
            else None
        )

    def test_inference_image_classification_head(self):
        model = MobileViTV2ForImageClassification.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.6336e00, -7.3204e-02, -5.1883e-01]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))

if __name__ == '__main__':
    unittest.main(argv=['', 'MobileViTV2ModelIntegrationTest.test_inference_image_classification_head'])
