# coding=utf-8
# Copyright 2023 HuggingFace Inc..
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


import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import List

import safetensors
from accelerate.utils import write_basic_config

from diffusers import DiffusionPipeline, UNet2DConditionModel

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


# These utils relate to ensuring the right error message is received when running scripts
class SubprocessCallException(Exception):
    pass


def run_command(command: List[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occurred while running `command`
    """
    try:
        p = subprocess.Popen(' '.join(command), stdout=subprocess.PIPE, bufsize=1, shell=True)
        for line in iter(p.stdout.readline, b''):
            print(line)
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ExamplesTestsAccelerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp()
        cls.configPath = os.path.join(cls._tmpdir, "default_config.yml")

        write_basic_config(save_location=cls.configPath)
        cls._launch_args = ["accelerate", "launch", "--config_file", cls.configPath]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls._tmpdir)

    def test_text_to_image(self):
        print("test_text_to_image start")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                train_text_to_image.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_text_to_image_checkpointing(self):
        print("test_text_to_image_checkpointing start")
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 5, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 5
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # check can run an intermediate checkpoint
            unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder="checkpoint-2/unet")
            pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, unet=unet, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 7 total steps resuming from checkpoint 4

            resume_run_args = f"""
                train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-4
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {
                    # no checkpoint-2 -> check old checkpoints do not exist
                    # check new checkpoints exist
                    "checkpoint-4",
                    "checkpoint-6",
                },
            )

    def test_text_to_image_checkpointing_use_ema(self):
        print("test_text_to_image_checkpointing_use_ema start")
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 5, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 5
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --use_ema
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # check can run an intermediate checkpoint
            unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder="checkpoint-2/unet")
            pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, unet=unet, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 7 total steps resuming from checkpoint 4

            resume_run_args = f"""
                train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-4
                --use_ema
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {
                    # no checkpoint-2 -> check old checkpoints do not exist
                    # check new checkpoints exist
                    "checkpoint-4",
                    "checkpoint-6",
                },
            )

    def test_text_to_image_checkpointing_checkpoints_total_limit(self):
        print("test_text_to_image_checkpointing_checkpoints_total_limit start")
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 7, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps 2
                --checkpoints_total_limit 2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_text_to_image_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        print("test_text_to_image_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints start")
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 9, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4, 6, 8

            initial_run_args = f"""
                train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 9
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps 2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            # resume and we should try to checkpoint at 10, where we'll have to remove
            # checkpoint-2 and checkpoint-4 instead of just a single previous checkpoint

            resume_run_args = f"""
                train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 11
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps 2
                --resume_from_checkpoint checkpoint-8
                --checkpoints_total_limit 3
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8", "checkpoint-10"},
            )

    def test_text_to_image_sdxl(self):
        print("test_text_to_image_sdxl start")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                train_text_to_image_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_text_to_image_lora_checkpointing_checkpoints_total_limit(self):
        print("test_text_to_image_lora_checkpointing_checkpoints_total_limit start")
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 7, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                train_text_to_image_lora.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps 2
                --checkpoints_total_limit 2
                --seed=0
                --num_validation_images 0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None
            )
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_text_to_image_lora_sdxl_checkpointing_checkpoints_total_limit(self):
        print("test_text_to_image_lora_sdxl_checkpointing_checkpoints_total_limit start")
        prompt = "a prompt"
        pipeline_path = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 7, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path {pipeline_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps 2
                --checkpoints_total_limit 2
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(pipeline_path)
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_text_to_image_lora_sdxl(self):
        print("test_text_to_image_lora_sdxl start")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

    def test_text_to_image_lora_sdxl_text_encoder_checkpointing_checkpoints_total_limit(self):
        print("test_text_to_image_lora_sdxl_text_encoder_checkpointing_checkpoints_total_limit start")
        prompt = "a prompt"
        pipeline_path = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 7, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path {pipeline_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --train_text_encoder
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(pipeline_path)
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_text_to_image_lora_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        print("test_text_to_image_lora_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints start")
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 9, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4, 6, 8

            initial_run_args = f"""
                train_text_to_image_lora.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 9
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                --num_validation_images=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None
            )
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            # resume and we should try to checkpoint at 10, where we'll have to remove
            # checkpoint-2 and checkpoint-4 instead of just a single previous checkpoint

            resume_run_args = f"""
                train_text_to_image_lora.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 11
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-8
                --checkpoints_total_limit=3
                --seed=0
                --num_validation_images=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            pipe = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None
            )
            pipe.load_lora_weights(tmpdir)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8", "checkpoint-10"},
            )

    def test_text_to_image_lora_sdxl_with_text_encoder(self):
        print("test_text_to_image_lora_sdxl_with_text_encoder start")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name lambdalabs/pokemon-blip-captions
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --train_text_encoder
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"unet"` or `"text_encoder"` or `"text_encoder_2"` in their names.
            keys = lora_state_dict.keys()
            starts_with_unet = all(
                k.startswith("unet") or k.startswith("text_encoder") or k.startswith("text_encoder_2") for k in keys
            )
            self.assertTrue(starts_with_unet)


if __name__ == '__main__':
    unittest.main(argv=[' ', 'ExamplesTestsAccelerate.test_text_to_image'])

