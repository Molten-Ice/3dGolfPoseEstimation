import torch
import torchvision
import numpy as np
import cv2
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random

rescale_height = 800

#Very hacky fix to make matplotlib work on my pc
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

repo_dir = "c:/Users/James/git/3dGolfPoseDetection/"
downloaded_dir = repo_dir + "downloaded-data/"
save_dir = downloaded_dir + "saved-labels/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(downloaded_dir):
    os.makedirs(downloaded_dir)

labelled_filenames = os.listdir(save_dir)
img_num = 1
for filename in labelled_filenames:
    img_num = max(img_num, int(filename.split("_")[0])+1)
print(f"Starting on img_num: {img_num}")

# if not os.path.exists(downloaded_dir+/golfdb-entire-image):

#download and unzip kaggle data, stored in downloaded-data
# !kaggle datasets download -d andrewmvd/3d-golf-swing-dataset
# https://www.kaggle.com/datasets/marcmarais/golfdb-entire-image


images_left_path = f'{downloaded_dir}/images_left.txt'
#check if exists
if not os.path.exists(images_left_path):
    golfdb_dir = downloaded_dir + 'golfdb/Swing_events/'
    golfdb_foldernames = os.listdir(golfdb_dir)

    image_paths = []
    for foldername in golfdb_foldernames:
        folder_path = golfdb_dir + foldername + "/"
        folder_image_paths = [folder_path + x for x in os.listdir(folder_path)]
        image_paths += folder_image_paths
    random.shuffle(image_paths)
    print(len(image_paths), image_paths[:2])

    with open(images_left_path, 'w') as f:
        f.write("\n".join(image_paths))

    with open(f'{downloaded_dir}/backup.txt', 'w') as f:
        f.write("\n".join(image_paths))

# img = cv2.imread(downloaded_dir + 'golfdb/Swing_events/Address/0.jpg')
# display(Image.fromarray(img[:,:,::-1]))

def label_img(img, club_coordinates):
    grip, club = club_coordinates

    labelled = img.copy()
    cv2.circle(labelled, grip, 5, (255,0,255), -1)
    cv2.circle(labelled, club, 5, (255,0,255), -1)
    cv2.line(labelled, club, grip, (255, 0, 0), 2)
    
    box_buffer = max(img.shape[:2])//20
    box_cords = [[max(min(grip[0], club[0])-box_buffer, 1), max(min(grip[1], club[1])-box_buffer, 1)], #top left cord
            [min(max(grip[0], club[0])+box_buffer, img.shape[0]-1), min(max(grip[1], club[1])+box_buffer, img.shape[1]-1)]] # bottom right cord
    cv2.rectangle(labelled, box_cords[0], box_cords[1], (0, 0, 255), 2)
    return labelled, grip, club, box_cords

def click_event_label(event, x, y, flags, params):
        # checking for left mouse clicks or right mouse clicks
        if event==cv2.EVENT_RBUTTONDOWN:
            global skip
            skip = True
        if event == cv2.EVENT_LBUTTONDOWN:
            global golf_club_coordinates
            golf_club_coordinates.append([int(x),int(y)])

def click_event_save(event, x, y, flags, params):
    # checking for left mouse clicks or right mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        global save
        save = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        global redo
        redo = True

loop = True
while loop:
    #really images will be filename and file will be loaded in first name
    #check if file exists
    with open(images_left_path, 'r') as fin:
        data = fin.read().splitlines(True)
    filepath = data[0][:-1] # removing \n at end
    img = cv2.imread(filepath)

    #### Rescale image
    h, w = img.shape[:2]
    r = rescale_height / float(h)
    dim = (int(w * r), rescale_height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    ### Labelled images in window
    golf_club_coordinates = []
    skip = False
        
    cv2.imshow('image', resized)
    cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback('image', click_event_label)
    while len(golf_club_coordinates) != 2:
        if skip:
            break
        k = cv2.waitKey(100)

        if k == 27:
            print("Ending loop.")
            loop = False
            cv2.destroyWindow('image') #make sure window closes cleanly
            break
    if not loop:
        break

    if skip:
        with open(images_left_path, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(images_left_path, 'w') as fout:
            fout.writelines(data[1:])
        continue

    ### Annotate labels on images
    labelled, grip, club, box_cords = label_img(resized, golf_club_coordinates)
    save = False
    redo = False

    cv2.imshow('image', labelled)
    cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback('image', click_event_save)
    while not (save or redo):
        k = cv2.waitKey(100)
        if k == 27:
            print("Ending loop.")
            loop = False
            cv2.destroyWindow('image') #make sure window closes cleanly
            break
    if not loop:
        break
    
    if save:
        save_filename = f"{img_num}_"
        for x, y in [grip] + [club]:
            save_filename += f"{x}-{y}-"
        save_filename = save_filename[:-1] +  "_.png"
        save_path = save_dir + save_filename
        cv2.imwrite(save_path, resized)

        img_num+=1
        with open(images_left_path, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(images_left_path, 'w') as fout:
            fout.writelines(data[1:])
    # else:
    #     print("redoing image")

print(f"Ending on img_num: {img_num}")