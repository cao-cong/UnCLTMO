import cv2
import os
import numpy as np
import glob
from PIL import ImageFont,ImageDraw,Image

scene_list = ['beerfest_lightshow_05','bistro_01','carousel_fireworks_07','Cleaning-3Exp-2Stop','exhibition_area_light','fireplace_01','fishing_longshot_01','fishing_longshot_02','fishing_longshot_03','ThrowingTowel-2Exp-3Stop']

for scene in scene_list:
    save_dir = 'video_results'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fps = 20  
    img_size = (1280,720)
    video_path = save_dir+'/{}.avi'.format(scene)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') #opencv3.0
    frame_paths = glob.glob('output_{}/{}/*.png'.format('ours', scene))
    frame_paths = sorted(frame_paths)
    

    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
    ref = cv2.imread(frame_paths[0]).astype(np.float32)
    exposure = np.mean(ref)
    for i in range(0,len(frame_paths)):
        print(frame_paths[i])
        result_frame = cv2.imread(frame_paths[i]).astype(np.float32)
        result_frame = result_frame*(exposure/np.mean(result_frame))
        result_frame = np.uint8(np.clip(result_frame,0,255))

        videoWriter.write(result_frame)

    videoWriter.release()