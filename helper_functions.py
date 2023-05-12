### Helper functions ###
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import subprocess as sp


valid_pairings = np.array([[16, 10], [15, 8], [14, 6], [13, 9], [12, 7], [11, 5], [6, 15], [5, 13], [4, 11], [3, 16], [2, 14], [1, 12]])  # [10/9, 0]
args_viz_subject = "output.mp4"
args_viz_action = "custom"
args_viz_camera = 0

# VideoPose3D keypoints -- red is right --
# [16 red wrist, 15 red elbow, 14 red shoulder, 13 black wrist, 12 black elbow, 11 black shoulder, 10 tip of head, 9 second point from head, 
# 8 middle of shoulder, 7 stomach, 6 black foot, 5 black knee, 4 black hip, 3 red foot, 2 red knee, 1 red hip, 0 middle of thighs
# detectron keypoints
# "keypoints": [ 0"nose", 1"left_eye", 2"right_eye", 3"left_ear", 4"right_ear", 5"left_shoulder",
# 6"right_shoulder", 7"left_elbow", 8"right_elbow", 9"left_wrist", 10"right_wrist", 11"left_hip",
# 12"right_hip", 13"left_knee", 14"right_knee", 15"left_ankle", 16"right_ankle" ]

def load_keypoints():

    print('Loading 2D detections...')
    keypoints = np.load('keypoints_2d.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoints = keypoints['positions_2d'].item()[args_viz_subject][args_viz_action][args_viz_camera]

    print('Loading 3D detections...')
    poses = np.load('keypoints_3d.npy')

    print('Loading 2D club...')
    club_keypoints_2d = np.load('club_keypoints_2d.npy')

    print(keypoints[0, :2], poses[0, :2], club_keypoints_2d[0])
    return keypoints, poses, club_keypoints_2d, keypoints_metadata, keypoints_symmetry

def find_median_sf(poses, keypoints):
    scale_factors = []
    for i in range(len(poses)):

        pos_unscaled = poses[i].copy()
        frame_2d_keypoints = keypoints[i].copy()

        known_3d = pos_unscaled[valid_pairings[:, 0]]
        differences = []
        for idx in range(len(known_3d)):
            differences.append(np.linalg.norm(known_3d - known_3d[idx], axis=1).mean())
        factor_3d = np.sort(differences)[len(differences)//2]

        known_2d = frame_2d_keypoints[valid_pairings[:, 1]]
        differences = []
        for idx in range(len(known_2d)):
            differences.append(np.linalg.norm(known_2d - known_2d[idx], axis=1).mean())
        factor_2d = np.sort(differences)[len(differences)//2]

        sf = factor_2d/factor_3d
        scale_factors.append(sf)
    scale_factors = np.sort(scale_factors)
    return scale_factors[len(scale_factors)//2]

def align_3d_skeleton(pos_unscaled, frame_2d_keypoints, sf, print_difference = False):

    pos = pos_unscaled*sf
    translations_differences = (frame_2d_keypoints[valid_pairings[:, 1]] - pos[valid_pairings[:, 0], :2])
    x_translation = np.sort(translations_differences[:, 0])[len(translations_differences)//2]
    y_translation = np.sort(translations_differences[:, 1])[len(translations_differences)//2]
    translation = [x_translation, y_translation]
    pos[:, :2] = pos[:, :2] + translation

    if print_difference:
        differences = pos[valid_pairings[:, 0], :2] -  frame_2d_keypoints[valid_pairings[:, 1]]
        print(f"Average difference: {abs(differences).mean():.2f} | translation: {translation} | sf: {sf:.2f}")

    return pos


### Only for generating coordinate graph ###
def graph_coordinates(club_coordinates_original, all_z_coordinates, club_coordinates):
    plt.clf()
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    def plot_3d_cords(ax, array, idx, title, show_z = False):
        ax.plot(list(range(len(array))), array[:, idx, 0], marker='o', color='blue', label='grip x',  markersize=2, linewidth=1)
        ax.plot(list(range(len(array))), array[:, idx, 1], marker='o', color='green', label='grip y',  markersize=2, linewidth=1)
        if show_z: ax.plot(list(range(len(array))), array[:, idx, 2], marker='o', color='red', label='grip z',  markersize=2, linewidth=1)
        ax.legend(loc="upper left")
        ax.set_title(title)
    plot_3d_cords(ax[0, 0], club_coordinates_original, idx=0, title="Original Grip", show_z=True) #Original grip
    plot_3d_cords(ax[1, 0], club_coordinates_original, idx=1, title="Original Head") #Original head
    plot_3d_cords(ax[0, 1], club_coordinates, idx=0,title="Translated Grip", show_z=True) #Translated grip
    plot_3d_cords(ax[1, 1], club_coordinates, idx=1,title="Translated Head") #Translated head


    ax[0, 2].plot(list(range(len(all_z_coordinates))), all_z_coordinates[:, 1], marker='o', color='blue', label='possible z1',  markersize=2, linewidth=1)
    ax[0, 2].plot(list(range(len(all_z_coordinates))), all_z_coordinates[:, 2], marker='o', color='green', label='possible z2',  markersize=2, linewidth=1)
    ax[0, 2].legend(loc="upper left")
    ax[0, 2].set_title("All possible z head coordinates")

    ax[1, 2].plot(list(range(len(club_coordinates))), club_coordinates[:, 1, 2], marker='o', color='red', label='possible z1',  markersize=2, linewidth=1)
    ax[1, 2].legend(loc="upper left")
    ax[1, 2].set_title("Z head coordinates used")

    plt.show()

def sphere_z_values(x, y, center, radius):
    x0, y0, z0 = center
    r_squared = radius ** 2

    # Calculate the term inside the square root
    term = r_squared - (x - x0) ** 2 - (y - y0) ** 2

    # Check if the term is non-negative, as square root of a negative number is not real
    if term >= 0:
        sqrt_term = term ** 0.5
        z1 = z0 + sqrt_term
        z2 = z0 - sqrt_term
        return z1, z2
    else:
        return None

def generate_aligned_coordinates(poses, keypoints, club_keypoints_2d, take_minimum = True):
    """Generate 3d predictions"""

    club_lengths_2d = sorted([np.linalg.norm(club[0] - club[1]) for club in club_keypoints_2d])
    radius = club_lengths_2d[-5] * 1 #orthogonal_factor
    sf = find_median_sf(poses, keypoints)
    print(f"Radius = {radius:.3f} | sf {sf:.3f}")
    adjusted_poses = []
    club_coordinates = []
    club_coordinates_original = []
    all_z_coordinates = []

    head = [0, 0, 0] # deals with case where it can't spot something in first few frames
    for i in range(len(poses)):
        pos_unscaled = poses[i].copy()
        frame_2d_keypoints = keypoints[i].copy()
        frame_club_2d_keypoints = club_keypoints_2d[i].copy()
        pos = align_3d_skeleton(pos_unscaled, frame_2d_keypoints, sf, print_difference = False)
        adjusted_poses.append(pos)

        average_wrist_x, average_wrist_y, average_wrist_z =  pos[[13, 16]].mean(axis=0)
        grip = np.array(list(frame_club_2d_keypoints[0]) + [average_wrist_z]) #use detectron2 x,y cords for 3d grip

        z_values = sphere_z_values(*frame_club_2d_keypoints[1], grip, radius)
        if z_values is not None:
            if take_minimum:
                head = list(frame_club_2d_keypoints[1]) + [min(z_values)]
            else:
                head = list(frame_club_2d_keypoints[1]) + [max(z_values)]
            all_z_coordinates.append([average_wrist_z] + list(z_values))
        else:
            all_z_coordinates.append(all_z_coordinates[-1])
        
        grip_new = np.array([average_wrist_x, average_wrist_y, average_wrist_z])
        adjusted_club = np.array([grip, head])
        club_coordinates_original.append(adjusted_club)
        
        translation_vector = grip_new - grip
        adjusted_club =  adjusted_club + translation_vector
        club_coordinates.append(adjusted_club)

    adjusted_poses = np.array(adjusted_poses)
    club_coordinates = np.array(club_coordinates)
    club_coordinates_original = np.array(club_coordinates_original)
    all_z_coordinates = np.array(all_z_coordinates)


    return adjusted_poses, club_coordinates, club_coordinates_original

def get_resolution(filename):
    """Returns height, width of video"""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)
            
def get_fps(filename):
    """Returns the fps of the video"""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)

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