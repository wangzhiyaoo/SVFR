"""
This script is based on the original project by https://huggingface.co/fffiloni.
URL: https://huggingface.co/spaces/fffiloni/SVFR-demo/blob/main/app.py

Modifications made:
- Synced the infer code updates from GitHub repo.
- Added an inpainting option to enhance functionality.

Author of modifications: https://github.com/wangzhiyaoo
Date: 2025/01/15
"""

import torch
import sys
import os
import subprocess
import shutil
import tempfile
import uuid
import gradio as gr
from glob import glob
from huggingface_hub import snapshot_download
import random

import argparse
import warnings
import os
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
import random

from omegaconf import OmegaConf
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection
import torchvision.transforms as transforms
import torch.nn.functional as F
from src.models.svfr_adapter.unet_3d_svd_condition_ip import UNet3DConditionSVDModel

# pipeline 
from src.pipelines.pipeline import LQ2VideoLongSVDPipeline

from src.utils.util import (
    save_videos_grid,
    seed_everything,
)
from torchvision.utils import save_image

from src.models.id_proj import IDProjConvModel
from src.models import model_insightface_360k

from src.dataset.face_align.align import AlignImage

warnings.filterwarnings("ignore")

import decord
import cv2
from src.dataset.dataset import get_affine_transform, mean_face_lm5p_256, get_union_bbox, process_bbox, crop_resize_img


# # Download models
# os.makedirs("models", exist_ok=True)

# snapshot_download(
#     repo_id = "fffiloni/SVFR",
#     local_dir = "./models"  
# )

# # List of subdirectories to create inside "checkpoints"
# subfolders = [
#     "stable-video-diffusion-img2vid-xt"
# ]
# # Create each subdirectory
# for subfolder in subfolders:
#     os.makedirs(os.path.join("models", subfolder), exist_ok=True)

# snapshot_download(
#     repo_id = "stabilityai/stable-video-diffusion-img2vid-xt",
#     local_dir = "./models/stable-video-diffusion-img2vid-xt"  
# )

BASE_DIR = '.'

config = OmegaConf.load("./config/infer.yaml")
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    f"{BASE_DIR}/{config.pretrained_model_name_or_path}", 
    subfolder="vae",
    variant="fp16")

val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
    f"{BASE_DIR}/{config.pretrained_model_name_or_path}", 
    subfolder="scheduler")

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    f"{BASE_DIR}/{config.pretrained_model_name_or_path}", 
    subfolder="image_encoder",
    variant="fp16")
unet = UNet3DConditionSVDModel.from_pretrained(
    f"{BASE_DIR}/{config.pretrained_model_name_or_path}", 
    subfolder="unet",
    variant="fp16")

weight_dir = 'models/face_align'
det_path = os.path.join(BASE_DIR, weight_dir, 'yoloface_v5m.pt')
align_instance = AlignImage("cuda", det_path=det_path)

to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

import torch.nn as nn
class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        x = super().forward(x)
        return x
# Add ref channel
old_weights = unet.conv_in.weight
old_bias = unet.conv_in.bias
new_conv1 = InflatedConv3d(
    12,
    old_weights.shape[0],
    kernel_size=unet.conv_in.kernel_size,
    stride=unet.conv_in.stride,
    padding=unet.conv_in.padding,
    bias=True if old_bias is not None else False,
)
param = torch.zeros((320, 4, 3, 3), requires_grad=True)
new_conv1.weight = torch.nn.Parameter(torch.cat((old_weights, param), dim=1))
if old_bias is not None:
    new_conv1.bias = old_bias
unet.conv_in = new_conv1
unet.config["in_channels"] = 12
unet.config.in_channels = 12


id_linear = IDProjConvModel(in_channels=512, out_channels=1024).to(device='cuda')

# load pretrained weights
unet_checkpoint_path = os.path.join(BASE_DIR, config.unet_checkpoint_path)
unet.load_state_dict(
    torch.load(unet_checkpoint_path, map_location="cpu"),
    strict=True,
)

id_linear_checkpoint_path = os.path.join(BASE_DIR, config.id_linear_checkpoint_path)
id_linear.load_state_dict(
    torch.load(id_linear_checkpoint_path, map_location="cpu"),
    strict=True,
)

net_arcface = model_insightface_360k.getarcface(f'{BASE_DIR}/{config.net_arcface_checkpoint_path}').eval().to(device="cuda")    

if config.weight_dtype == "fp16":
    weight_dtype = torch.float16
elif config.weight_dtype == "fp32":
    weight_dtype = torch.float32
elif config.weight_dtype == "bf16":
    weight_dtype = torch.bfloat16
else:
    raise ValueError(
        f"Do not support weight dtype: {config.weight_dtype} during training"
    )

image_encoder.to(weight_dtype)
vae.to(weight_dtype)
unet.to(weight_dtype)
id_linear.to(weight_dtype)
net_arcface.requires_grad_(False).to(weight_dtype) 

pipe = LQ2VideoLongSVDPipeline(
    unet=unet,
    image_encoder=image_encoder,
    vae=vae,
    scheduler=val_noise_scheduler,
    feature_extractor=None

)
pipe = pipe.to("cuda", dtype=unet.dtype)

def gen(args,pipe):
    save_dir = f"{BASE_DIR}/{args.output_dir}"
    os.makedirs(save_dir,exist_ok=True)

    seed_input = args.seed
    seed_everything(seed_input)

    video_path = args.input_path
    task_ids = args.task_ids
    
    if 2 in task_ids and args.mask_path is not None: 
        mask_path = args.mask_path
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)

        white_positions = mask_array == 255

    print('task_ids:',task_ids)
    task_prompt = [0,0,0]
    for i in range(3):
        if i in task_ids:
            task_prompt[i] = 1
    print("task_prompt:",task_prompt)
    
    video_name = video_path.split('/')[-1]
    # print(video_name)

    if os.path.exists(os.path.join(save_dir, "result_frames", video_name[:-4])):
        print(os.path.join(save_dir, "result_frames", video_name[:-4]))
        # continue

    cap = decord.VideoReader(video_path, fault_tol=1)
    total_frames = len(cap)
    T = total_frames #
    print("total_frames:",total_frames)
    step=1
    drive_idx_start = 0
    drive_idx_list = list(range(drive_idx_start, drive_idx_start + T * step, step))
    assert len(drive_idx_list) == T

    # Crop faces from the video for further processing
    bbox_list = []
    frame_interval = 5
    for frame_count, drive_idx in enumerate(drive_idx_list):
        if frame_count % frame_interval != 0:
            continue  
        frame = cap[drive_idx].asnumpy()
        _, _, bboxes_list = align_instance(frame[:,:,[2,1,0]], maxface=True)
        if bboxes_list==[]:
            continue
        x1, y1, ww, hh = bboxes_list[0]
        x2, y2 = x1 + ww, y1 + hh
        bbox = [x1, y1, x2, y2]
        bbox_list.append(bbox)
    bbox = get_union_bbox(bbox_list)
    bbox_s = process_bbox(bbox, expand_radio=0.4, height=frame.shape[0], width=frame.shape[1])

    imSameIDs = []
    vid_gt = []
    for i, drive_idx in enumerate(drive_idx_list):
        frame = cap[drive_idx].asnumpy()
        imSameID = Image.fromarray(frame)
        imSameID = crop_resize_img(imSameID, bbox_s)
        imSameID = imSameID.resize((512,512))
        if 1 in task_ids:
            imSameID = imSameID.convert("L")  # Convert to grayscale
            imSameID = imSameID.convert("RGB")
        image_array = np.array(imSameID)
        if 2 in task_ids and args.mask_path is not None:
            image_array[white_positions] = [255, 255, 255] # mask for inpainting task
        vid_gt.append(np.float32(image_array/255.))
        imSameIDs.append(imSameID)

    vid_lq = [(torch.from_numpy(frame).permute(2,0,1) - 0.5) / 0.5 for frame in vid_gt]

    val_data = dict(
        pixel_values_vid_lq = torch.stack(vid_lq,dim=0),
        # pixel_values_ref_img=self.to_tensor(target_image),
        # pixel_values_ref_concat_img=self.to_tensor(imSrc2),
        task_ids=task_ids,
        task_id_input=torch.tensor(task_prompt),
        total_frames=total_frames,
    )
    
    window_overlap=0
    inter_frame_list = get_overlap_slide_window_indices(val_data["total_frames"],config.data.n_sample_frames,window_overlap)
    
    lq_frames = val_data["pixel_values_vid_lq"]
    task_ids = val_data["task_ids"]
    task_id_input = val_data["task_id_input"]
    height, width = val_data["pixel_values_vid_lq"].shape[-2:]
    
    print("Generating the first clip...")
    output = pipe(
        lq_frames[inter_frame_list[0]].to("cuda").to(weight_dtype), # lq
        None, # ref concat
        torch.zeros((1, len(inter_frame_list[0]), 49, 1024)).to("cuda").to(weight_dtype),# encoder_hidden_states
        task_id_input.to("cuda").to(weight_dtype),
        height=height,
        width=width,
        num_frames=len(inter_frame_list[0]),
        decode_chunk_size=config.decode_chunk_size,
        noise_aug_strength=config.noise_aug_strength,
        min_guidance_scale=config.min_appearance_guidance_scale, 
        max_guidance_scale=config.max_appearance_guidance_scale,
        overlap=config.overlap,
        frames_per_batch=len(inter_frame_list[0]),
        num_inference_steps=50,
        i2i_noise_strength=config.i2i_noise_strength,
    )
    video = output.frames
 
    ref_img_tensor = video[0][:,-1]
    ref_img = (video[0][:,-1] *0.5+0.5).clamp(0,1) * 255.
    ref_img = ref_img.permute(1,2,0).cpu().numpy().astype(np.uint8)

    pts5 = align_instance(ref_img[:,:,[2,1,0]], maxface=True)[0][0]

    warp_mat = get_affine_transform(pts5, mean_face_lm5p_256 * height/256)
    ref_img = cv2.warpAffine(np.array(Image.fromarray(ref_img)), warp_mat, (height, width), flags=cv2.INTER_CUBIC)
    ref_img = to_tensor(ref_img).to("cuda").to(weight_dtype)
    
    save_image(ref_img*0.5 + 0.5,f"{save_dir}/ref_img_align.png")
    
    ref_img =  F.interpolate(ref_img.unsqueeze(0)[:, :, 0:224, 16:240], size=[112, 112], mode='bilinear')
    _, id_feature_conv = net_arcface(ref_img) 
    id_embedding = id_linear(id_feature_conv) 
    
    print('Generating all video clips...')
    video = pipe(
        lq_frames.to("cuda").to(weight_dtype), # lq
        ref_img_tensor.to("cuda").to(weight_dtype),
        id_embedding.unsqueeze(1).repeat(1, len(lq_frames), 1, 1).to("cuda").to(weight_dtype), # encoder_hidden_states
        task_id_input.to("cuda").to(weight_dtype),
        height=height,
        width=width,
        num_frames=val_data["total_frames"],#frame_num,
        decode_chunk_size=config.decode_chunk_size,
        noise_aug_strength=config.noise_aug_strength,
        min_guidance_scale=config.min_appearance_guidance_scale,
        max_guidance_scale=config.max_appearance_guidance_scale,
        overlap=config.overlap,
        frames_per_batch=config.data.n_sample_frames,
        num_inference_steps=config.num_inference_steps,
        i2i_noise_strength=config.i2i_noise_strength,
    ).frames


    video = (video*0.5 + 0.5).clamp(0, 1)
    video = torch.cat([video.to(device="cuda")], dim=0).cpu()
    save_videos_grid(video, f"{save_dir}/{video_name[:-4]}_{seed_input}_gen.mp4", n_rows=1, fps=25)

    lq_frames = lq_frames.permute(1,0,2,3).unsqueeze(0)
    lq_frames = (lq_frames * 0.5 + 0.5).clamp(0, 1).to(device="cuda").cpu()
    save_videos_grid(lq_frames, f"{save_dir}/{video_name[:-4]}_{seed_input}_ori.mp4", n_rows=1, fps=25)
    
    if args.restore_frames:
        video = video.squeeze(0)
        os.makedirs(os.path.join(save_dir, "result_frames", f"{video_name[:-4]}_{seed_input}"),exist_ok=True)
        print(os.path.join(save_dir, "result_frames", video_name[:-4]))
        for i in range(video.shape[1]):
            save_frames_path = os.path.join(f"{save_dir}/result_frames", f"{video_name[:-4]}_{seed_input}", f'{i:08d}.png')
            save_image(video[:,i], save_frames_path)


def get_overlap_slide_window_indices(video_length, window_size, window_overlap):
    inter_frame_list = []
    for j in range(0, video_length, window_size-window_overlap):
        inter_frame_list.append( [e % video_length for e in range(j, min(j + window_size, video_length))] )

    return inter_frame_list



def random_seed():
    return random.randint(0, 10000)

def infer(lq_sequence, task_name, mask, seed):
    
    unique_id = str(uuid.uuid4())
    output_dir = f"results_{unique_id}"

    task_mapping = {
        "BFR": 0,
        "Colorization": 1,
        "Inpainting": 2
    }
    
    task_ids = [task_mapping[task] for task in task_name if task in task_mapping]
    # task_id = ",".join(task_ids)
    
    try:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.task_ids = task_ids
        args.input_path = f"{lq_sequence}"
        args.output_dir = f"{output_dir}"
        args.mask_path = f"{mask}"
        args.seed = int(seed)
        args.restore_frames = False

        gen(args,pipe)

        # Search for the mp4 file in a subfolder of output_dir
        output_video = glob(os.path.join(output_dir,"*gen.mp4"))
        face_region_video = glob(os.path.join(output_dir,"*ori.mp4"))
        # print(face_region_video,output_video)
        
        if output_video:
            output_video_path = output_video[0]  # Get the first match
            face_region_video_path = face_region_video[0]  # Get the first match
        else:
            output_video_path = None
            face_region_video = None
        
        print(output_video_path,face_region_video_path)
        torch.cuda.empty_cache()
        return face_region_video_path,output_video_path
    
    except subprocess.CalledProcessError as e:
        torch.cuda.empty_cache()
        raise gr.Error(f"Error during inference: {str(e)}")

css="""
div#col-container{
    margin: 0 auto;
    max-width: 982px;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# SVFR: A Unified Framework for Generalized Video Face Restoration")
        gr.Markdown("SVFR is a unified framework for face video restoration that supports tasks such as BFR, Colorization, Inpainting, and their combinations within one cohesive system.")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/wangzhiyaoo/SVFR">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://wangzhiyaoo.github.io/SVFR/">
                <img src='https://img.shields.io/badge/Project-Page-green'>
            </a>
            <a href="https://arxiv.org/pdf/2501.01235">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                input_seq = gr.Video(label="Video LQ")
                task_name = gr.CheckboxGroup(
                    label="Task", 
                    choices=["BFR", "Colorization", "Inpainting"], 
                    value=["BFR"]  # default
                )
                mask_input = gr.Image(type="filepath",label="Inpainting Mask")
                with gr.Row():
                    seed_input = gr.Number(label="Seed", value=77, precision=0)
                    random_seed_btn = gr.Button("🎲",scale=1,elem_id="dice-btn")  
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear")
            with gr.Column():
                output_face = gr.Video(label="Face Region Input")
                output_res = gr.Video(label="Restored")
                gr.Examples(
                    examples = [
                        ["./assert/lq/lq1.mp4", ["BFR"],None],
                        ["./assert/lq/lq2.mp4", ["BFR", "Colorization"],None],
                        ["./assert/lq/lq3.mp4", ["BFR", "Colorization", "Inpainting"],"./assert/mask/lq3.png"]
                    ],
                    inputs = [input_seq, task_name, mask_input]
                )

    random_seed_btn.click(
        fn=random_seed, 
        inputs=[], 
        outputs=seed_input
    )


    submit_btn.click(
        fn = infer,
        inputs = [input_seq, task_name, mask_input,seed_input],
        outputs = [output_face,output_res]
    )
    clear_btn.click(
        fn=lambda: [None,["BFR"],None,77,None,None], 
        inputs=None,
        outputs=[input_seq, task_name, mask_input, seed_input, output_face, output_res]
    )

demo.queue().launch(show_api=False, show_error=True, server_port=1203)