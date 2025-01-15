import os
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import CLIPImageProcessor
# import librosa

import os
import cv2

mean_face_lm5p_256 = np.array([
[(30.2946+8)*2+16, 51.6963*2], 
[(65.5318+8)*2+16, 51.5014*2], 
[(48.0252+8)*2+16, 71.7366*2], 
[(33.5493+8)*2+16, 92.3655*2],  
[(62.7299+8)*2+16, 92.2041*2],
], dtype=np.float32)

def get_affine_transform(target_face_lm5p, mean_lm5p):
    mat_warp = np.zeros((2,3))
    A = np.zeros((4,4))
    B = np.zeros((4))
    for i in range(5):
        A[0][0] += target_face_lm5p[i][0] * target_face_lm5p[i][0] + target_face_lm5p[i][1] * target_face_lm5p[i][1]
        A[0][2] += target_face_lm5p[i][0]
        A[0][3] += target_face_lm5p[i][1]

        B[0] += target_face_lm5p[i][0] * mean_lm5p[i][0] + target_face_lm5p[i][1] * mean_lm5p[i][1]        #sb[1] += a[i].x*b[i].y - a[i].y*b[i].x;
        B[1] += target_face_lm5p[i][0] * mean_lm5p[i][1] - target_face_lm5p[i][1] * mean_lm5p[i][0]
        B[2] += mean_lm5p[i][0]
        B[3] += mean_lm5p[i][1]

    A[1][1] = A[0][0]
    A[2][1] = A[1][2] = -A[0][3]
    A[3][1] = A[1][3] = A[2][0] = A[0][2]
    A[2][2] = A[3][3] = 5
    A[3][0] = A[0][3]

    _, mat23 = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    mat_warp[0][0] = mat23[0]
    mat_warp[1][1] = mat23[0]
    mat_warp[0][1] = -mat23[1]
    mat_warp[1][0] = mat23[1]
    mat_warp[0][2] = mat23[2]
    mat_warp[1][2] = mat23[3]

    return mat_warp

def get_union_bbox(bboxes):
    bboxes = np.array(bboxes)
    min_x = np.min(bboxes[:, 0])
    min_y = np.min(bboxes[:, 1])
    max_x = np.max(bboxes[:, 2])
    max_y = np.max(bboxes[:, 3])
    return np.array([min_x, min_y, max_x, max_y])


def process_bbox(bbox, expand_radio, height, width):
    
    def expand(bbox, ratio, height, width):
        
        bbox_h = bbox[3] - bbox[1]
        bbox_w = bbox[2] - bbox[0]
        
        expand_x1 = max(bbox[0] - ratio * bbox_w, 0)
        expand_y1 = max(bbox[1] - ratio * bbox_h, 0)
        expand_x2 = min(bbox[2] + ratio * bbox_w, width)
        expand_y2 = min(bbox[3] + ratio * bbox_h, height)

        return [expand_x1,expand_y1,expand_x2,expand_y2]

    def to_square(bbox_src, bbox_expend, height, width):

        h = bbox_expend[3] - bbox_expend[1]
        w = bbox_expend[2] - bbox_expend[0]
        c_h = (bbox_expend[1] + bbox_expend[3]) / 2
        c_w = (bbox_expend[0] + bbox_expend[2]) / 2

        c = min(h, w) / 2

        c_src_h = (bbox_src[1] + bbox_src[3]) / 2
        c_src_w = (bbox_src[0] + bbox_src[2]) / 2

        s_h, s_w = 0, 0
        if w < h:
            d = abs((h - w) / 2)
            s_h = min(d, abs(c_src_h-c_h))
            s_h = s_h if  c_src_h > c_h else s_h * (-1)
        else:
            d = abs((h - w) / 2)
            s_w = min(d, abs(c_src_w-c_w))
            s_w = s_w if  c_src_w > c_w else s_w * (-1)


        c_h = (bbox_expend[1] + bbox_expend[3]) / 2 + s_h
        c_w = (bbox_expend[0] + bbox_expend[2]) / 2 + s_w

        square_x1 = c_w - c
        square_y1 = c_h - c
        square_x2 = c_w + c
        square_y2 = c_h + c 

        return [round(square_x1), round(square_y1), round(square_x2), round(square_y2)]


    bbox_expend = expand(bbox, expand_radio, height=height, width=width)
    processed_bbox = to_square(bbox, bbox_expend, height=height, width=width)

    return processed_bbox


def crop_resize_img(img, bbox):
    x1, y1, x2, y2 = bbox
    img = img.crop((x1, y1, x2, y2))
    return img
