import os
from glob import glob
import subprocess

# recalc_classes = ['Assault', 'Fight']

def dataset_jpg(dataset_path, jpg_path):
    '''
    converts all files to jpg format
    '''
    if not os.path.exists(jpg_path):
        os.mkdir(jpg_path)
    inner_files = glob(os.path.join(dataset_path, '*'))
    dataset_folders = [name for name in inner_files if os.path.isdir(name) and os.path.split(name)[-1].istitle()] # filter out files and non Title names
    for class_folder in dataset_folders:
        curr_class = os.path.split(class_folder)[-1]
        # if curr_class not in recalc_classes:
        #     continue
        os.mkdir(os.path.join(jpg_path, curr_class))
        print(os.path.join(jpg_path, curr_class), 'created')
        scene_folders = glob(os.path.join(class_folder, '*'))
        scene_folders = [scene_folder for scene_folder in scene_folders if '.DS' not in scene_folder]
        for scene_folder in scene_folders:
            curr_scene = os.path.split(scene_folder)[-1]
            os.mkdir(os.path.join(jpg_path, curr_class, curr_scene))
            print(os.path.join(jpg_path, curr_class, curr_scene), 'created')
            scene_videos = glob(os.path.join(scene_folder, '*.mp4'))
            for video in scene_videos:
                # create folder
                video_name = os.path.split(video)[-1][:-4]
                dest_path = os.path.join(jpg_path, curr_class, curr_scene, video_name)
                os.mkdir(dest_path)
                print(dest_path, 'created')
                cmd = ['ffmpeg', '-i', str(video), '-vf', 'scale=-1:360', str(video) + '/image_%05d.jpg']
                # cmd = 'ffmpeg -i "{}" -vf scale=-1:360 "{}/image_%05d.jpg"'.format(video, dest_path)
                print(cmd)
                subprocess.call(cmd, shell=True)
                print(video, 'done!')
                print('\n')