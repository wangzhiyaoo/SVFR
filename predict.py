# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from cog import BasePredictor, Input, Path
from argparse import Namespace
from typing import List

import torch
from omegaconf import OmegaConf
import gdown  # Add gdown to requirements.txt

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
        import gdown

        # 1. Load config
        config_path = "./config/infer.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        self.config = OmegaConf.load(config_path)

        try:
            # Add download logic at the start of setup
            for model_file in MODEL_FILES:
                url = f"{WEIGHTS_BASE_URL}/{MODEL_CACHE}/{model_file}"
                dest_path = f"{MODEL_CACHE}/{model_file}"

                dir_name = dest_path.replace(".tar", "")
                if os.path.exists(dir_name):
                    print(f"[+] Directory {dir_name} already exists, skipping download")
                    continue

                download_weights(url, dest_path)
        except (Exception, KeyboardInterrupt) as e:
            print(f"[ERROR] Failed to download weights: {e}")
            # raise  # Re-raise the exception to properly handle interrupts

            # 2. Create model directories
            models_dir = "./models"
            os.makedirs(f"{models_dir}/face_align", exist_ok=True)
            os.makedirs(f"{models_dir}/face_restoration", exist_ok=True)

            # 3. Download SVFR models (from Google Drive)
            gdrive_url = "https://drive.google.com/drive/folders/1nzy9Vk-yA_DwXm1Pm4dyE2o0r7V6_5mn"
            model_files = {
                f"{models_dir}/face_align/yoloface_v5m.pt": "Face detection model (YOLO)",
                f"{models_dir}/face_restoration/unet.pth": "UNet restoration model",
                f"{models_dir}/face_restoration/id_linear.pth": "ID projection model",
                f"{models_dir}/face_restoration/insightface_glint360k.pth": "InsightFace model",
            }

            # NOTE: models/stable-video-diffusion-img2vid-xt is gotten via git lfs (see readme)

            missing_files = []
            for file_path, description in model_files.items():
                if not os.path.exists(file_path):
                    print(f"Downloading {description}...")
                    try:
                        # Download the entire folder once
                        if not hasattr(self, "_gdrive_downloaded"):
                            gdown.download_folder(
                                gdrive_url, output=models_dir, quiet=False
                            )
                            self._gdrive_downloaded = True
                            break
                    except Exception as e:
                        missing_files.append(f"• {description} ({file_path}): {str(e)}")

            # 4. Verify all required files exist
            missing_files = []
            for file_path, description in model_files.items():
                if not os.path.exists(file_path):
                    missing_files.append(f"• {description} ({file_path})")

            if missing_files:
                raise FileNotFoundError(
                    "Missing required model files:\n"
                    + "\n".join(missing_files)
                    + "\nPlease ensure you have access to the Google Drive folder and try again."
                )

    def predict(
        self,
        video: Path = Input(
            description="Input video file that will be enhanced, supporting common formats like MP4.",
        ),
        tasks: str = Input(
            choices=[
                "face-restoration",
                "face-restoration-and-colorization",
                "face-restoration-and-colorization-and-inpainting",
            ],
            default="face-restoration",
            description="Select which restoration tasks to apply, where face-restoration enhances facial details only, face-restoration-and-colorization enhances faces and restores colors, and face-restoration-and-colorization-and-inpainting provides the full pipeline requiring a mask input.",
        ),
        mask: Path = Input(
            default=None,
            description="An inpainting mask image where white areas indicate regions that will be restored, which is only required when using the full pipeline with inpainting.",
        ),
        seed: int = Input(
            description="Random seed. Use -1 to randomize the seed", default=-1
        ),
    ) -> Path:
        """Run face restoration pipeline with selected enhancements"""

        if seed == -1:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Convert friendly task names to the task IDs that infer.py expects
        task_map = {
            "face-restoration": [0],
            "face-restoration-and-colorization": [0, 1],
            "face-restoration-and-colorization-and-inpainting": [0, 1, 2],
        }
        task_ids = task_map[tasks]

        # Validate inpainting
        if "face-restoration-and-colorization-and-inpainting" == tasks and mask is None:
            raise ValueError(
                "When using the full pipeline with inpainting, you must provide a mask image. "
                "The mask should be black & white where white pixels indicate areas to restore."
            )

        # Create output directory
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # Run pipeline
        args = Namespace(
            config="./config/infer.yaml",
            output_dir=output_dir,
            seed=seed,
            task_ids=task_ids,
            input_path=str(video),
            mask_path=str(mask) if mask else None,
            restore_frames=False,
        )

        main(self.config, args)

        # Build output path
        base_name = os.path.splitext(os.path.basename(str(video)))[0]
        out_path = os.path.join(output_dir, f"{base_name}_{seed}.mp4")

        if not os.path.exists(out_path):
            raise RuntimeError(
                f"Expected output video not found at: {out_path}\n"
                "Check logs for potential errors."
            )

        return Path(out_path)
