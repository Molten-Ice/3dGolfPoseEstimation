import os
import shutil
import subprocess
 
os.makedirs('VideoPose3D/inference/videos/')
shutil.copyfile('output.mp4', 'VideoPose3D/inference/videos/output.mp4')

#inferring 2D keypoints with Detectron, creatings VideoPose3D/inference/output_directory/output.mp4.npz
import os
os.mkdir("VideoPose3D/inference/output_directory")

result = subprocess.run(f'cd VideoPose3D/inference/ && python infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir output_directory \
    --image-ext mp4 \
    videos', shell=True, capture_output=True, text=True)
print(result)
#Move output_directory from /inference to /data
import shutil 
shutil.move('VideoPose3D/inference/output_directory', 'VideoPose3D/data/output_directory') 

#Create custom dataset VideoPose3D/data/data_2d_custom_myvideos.npz, using .npz files in VideoPose3D/data/output_directory
result = subprocess.run(f'cd VideoPose3D/data/ && python prepare_data_2d_custom.py -i output_directory -o myvideos', shell=True, capture_output=True, text=True)
print(result)

#Generate and save 3d keypoints
result = subprocess.run(f'cd VideoPose3D/ && python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject output.mp4 --viz-action custom --viz-camera 0 --viz-video inference/videos/output.mp4 --viz-export keypoints_3d --viz-size 6', shell=True, capture_output=True, text=True)
print(result)

shutil.move('VideoPose3D/data/data_2d_custom_myvideos.npz', 'keypoints_2d.npz')
shutil.move('VideoPose3D/keypoints_3d.npy', 'keypoints_3d.npy')

print("----- 3d keypoints generated!! ------")

