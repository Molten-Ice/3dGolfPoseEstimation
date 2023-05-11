### 2d inference ###

api_key_path = 'kaggle_apikey.json'
video_path = 'output.mp4'
repo_dir = "/content/"

import json
with open(api_key_path, 'r') as json_file:
    kaggle_apikey = json.load(json_file)

import numpy as np
import subprocess as sp
def get_resolution(filename):
    
    """Returns height, width of video"""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)
            

def read_video(filename, skip=0, limit=-1):
    """This function reads a video file and yields each frame as a numpy array in RGB format"""
    w, h = get_resolution(filename)
    
    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']
    
    i = 0
    with sp.Popen(command, stdout = sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w*h*3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))

import torch
import torchvision
import torch.nn as nn
import albumentations as A # Library for augmentations
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

import os
import cv2
import json
import time
import shutil
import random
import zipfile
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

data_path = repo_dir + "downloaded-data/"



device = 'cuda:0'
# loaded_model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
loaded_model = keypointrcnn_resnet50_fpn()
loaded_model.eval()
out = nn.ConvTranspose2d(512, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
loaded_model.roi_heads.keypoint_predictor.kps_score_lowres = out
loaded_model.load_state_dict(torch.load('downloaded-data/'+model_name))
loaded_model = loaded_model.to(device)
loaded_model.eval()
# print(list(loaded_model.backbone.fpn.parameters())[0][:5, :5, 0, 0])
print("Loaded model")

class GolfDataset(torch.utils.data.Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.img_transforms = transforms.ToTensor()
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        img = self.frames[idx]
        target = {}
        # target["boxes"] = torch.Tensor().unsqueeze(dim=0).type(torch.FloatTensor)
        # target["labels"] = torch.Tensor(torch.ones(1)).type(torch.int64)
        # target["keypoints"] = torch.Tensor([grip + [1.], head + [1.]]).unsqueeze(dim=0).type(torch.FloatTensor)

        img = Image.fromarray(img)
        img = self.img_transforms(img)
        return img, target
    
def collate_fn(batch):
    return tuple(zip(*batch))

def generate_cords(frames):
    global keypoints
    global boxes
    dataset = GolfDataset(frames)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    print(f"Dataset lengths: {len(dataset)}| Dataloader #batches: {len(dataloader)}, batches of size {batch_size}")

    coordinates = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            preds = loaded_model([img.to(device) for img in batch[0]])
            for labels in preds:
                try:
                    grip = [int(x) for x in labels['keypoints'][0, 0, :2]]
                    head = [int(x) for x in labels['keypoints'][0, 1, :2]]
                    box = [int(x) for x in labels['boxes'][0]]
                    keypoints_for_frame = [grip, head] # don't want to save box here
                    box_for_frame = box
                except Exception as e:
                    print(f"Skipping  frame | {e}")
                    keypoints_for_frame = keypoints[-1]
                    box_for_frame = boxes[-1]
                keypoints.append(keypoints_for_frame)
                boxes.append(box_for_frame)

# Load video using ffmpeg
batch_size = 2
frames_at_once = 20
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

keypoints = []
boxes = []

frame_idx = -1
frames = []
loaded_model.eval()
for f in read_video(video_path):
    frame_idx+=1
    # if frame_idx < 192:
    #     keypoints.append([[0,0], [0,0]])
    #     boxes.append([0,0,0,0])
    #     continue
    frames.append(f)

    #split image into segments of size frames_at_once so it doesn't have to all be loaded at once
    if (len(frames) % frames_at_once == 0 and len(frames) != 0):
        print(f"Generating keypoints: {frame_idx}")
        generate_cords(frames)
        frames = []
        
if (len(frames) % frames_at_once != 0):
    print(f"Generating keypoints: {frame_idx}")
    generate_cords(frames)


np.save("club_keypoints_2d", keypoints)
print("Saved club_keypoints_2d.npy")
# np.save("boxes", boxes)