# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from cog import BasePredictor, Input, Path
from argparse import Namespace

from omegaconf import OmegaConf

from infer import main

MODEL_CACHE = "models"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE
WEIGHTS_BASE_URL = f"https://weights.replicate.delivery/default/SVFR"
MODEL_FILES = [
    "face_align.tar",
    "face_restoration.tar",
    "stable-video-diffusion-img2vid-xt.tar",
]


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)

    if ".tar" in dest:
        dest = os.path.dirname(dest)

    command = ["pget", "-vfx", url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Load models and prepare environment on startup
        """

        # 1. Load config
        config_path = "./config/infer.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        self.config = OmegaConf.load(config_path)

        # Add download logic at the start of setup
        for model_file in MODEL_FILES:
            url = f"{WEIGHTS_BASE_URL}/{MODEL_CACHE}/{model_file}"
            dest_path = f"{MODEL_CACHE}/{model_file}"

            dir_name = dest_path.replace(".tar", "")
            if os.path.exists(dir_name):
                print(f"[+] Directory {dir_name} already exists, skipping download")
                continue

            download_weights(url, dest_path)

    def predict(
        self,
        video: Path = Input(description="Input video file (e.g. MP4)."),
        tasks: str = Input(
            choices=[
                "face-restoration",
                "face-restoration-and-colorization",
                "face-restoration-and-colorization-and-inpainting",
            ],
            default="face-restoration",
            description="Which restoration tasks to apply.",
        ),
        mask: Path = Input(
            default=None,
            description="An inpainting mask image (white areas will be restored). Only required when tasks includes inpainting.",
        ),
        # Below are the overrides matching default values in infer.yaml
        num_inference_steps: int = Input(
            default=30, description="Number of diffusion steps."
        ),
        decode_chunk_size: int = Input(
            default=16, description="Chunk size for decoding long videos."
        ),
        overlap: int = Input(
            default=3, description="Number of overlapping frames between segments."
        ),
        noise_aug_strength: float = Input(
            default=0.0, description="Noise augmentation strength."
        ),
        min_appearance_guidance_scale: float = Input(
            default=2.0, description="Minimum guidance scale for restoration."
        ),
        max_appearance_guidance_scale: float = Input(
            default=2.0, description="Maximum guidance scale for restoration."
        ),
        i2i_noise_strength: float = Input(
            default=1.0, description="Image-to-image noise strength."
        ),
        seed: int = Input(
            default=None, description="Random seed. Leave blank to randomize."
        ),
    ) -> Path:
        """Run face restoration pipeline with the selected enhancements."""

        # Handle random seed
        if seed == -1 or seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # If a user-supplied value differs from None (or the original code used None),
        # here we always override self.config with the new value.
        self.config.num_inference_steps = num_inference_steps
        self.config.decode_chunk_size = decode_chunk_size
        self.config.overlap = overlap
        self.config.noise_aug_strength = noise_aug_strength
        self.config.min_appearance_guidance_scale = min_appearance_guidance_scale
        self.config.max_appearance_guidance_scale = max_appearance_guidance_scale
        self.config.i2i_noise_strength = i2i_noise_strength

        # Convert REST-friendly tasks into internal numeric IDs
        task_map = {
            "face-restoration": [0],
            "face-restoration-and-colorization": [0, 1],
            "face-restoration-and-colorization-and-inpainting": [0, 1, 2],
        }
        task_ids = task_map[tasks]

        # If the user picks the inpainting task, enforce that they supply a mask
        if "face-restoration-and-colorization-and-inpainting" == tasks and mask is None:
            raise ValueError(
                "For inpainting, a mask image must be provided. (White areas are restored.)"
            )

        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # Build the arguments that get passed to infer.py
        args = Namespace(
            config="./config/infer.yaml",
            output_dir=output_dir,
            seed=seed,
            task_ids=task_ids,
            input_path=str(video),
            mask_path=str(mask) if mask else None,
            restore_frames=False,  # change to True if storing frames
        )

        main(self.config, args)

        # Construct final output path
        base_name = os.path.splitext(os.path.basename(str(video)))[0]
        out_path = os.path.join(output_dir, f"{base_name}_{seed}.mp4")

        if not os.path.exists(out_path):
            raise RuntimeError(
                f"Expected output video not found at: {out_path}\n"
                "Check logs for potential errors."
            )

        return Path(out_path)
