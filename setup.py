import subprocess
result = subprocess.run(f'git clone https://github.com/facebookresearch/VideoPose3D.git', shell=True, capture_output=True, text=True)
print("git clone VideoPose3D:", result)
result = subprocess.run(f'wget -P VideoPose3D/checkpoint/ https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin', shell=True, capture_output=True, text=True)
print("download 3d model:", result)