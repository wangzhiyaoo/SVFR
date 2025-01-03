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
