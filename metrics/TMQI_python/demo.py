import numpy as np
import cv2
import os
import glob
from TMQI import TMQI, TMQIr

avg_score = 0
num = 0
source_dir = '../../test_HDRvideo'
sub_dirs = os.listdir(source_dir)
for sub_dir in sub_dirs:
    folder_path = os.path.join(source_dir, sub_dir)
    #print(folder_path)
    hdr_paths = glob.glob(folder_path+'/*.npy')
    #print(hdr_paths)
    hdr_paths = sorted(hdr_paths)[:6]
    for hdr_path in hdr_paths:
        print(hdr_path)
        hdr = np.load(hdr_path).astype(np.float32)
        ldr_path = hdr_path.replace('test_HDRvideo', 'output_UnCLTMO')
        ldr_path = ldr_path.replace('.npy', '_UnCLTMO.png')
        ldr = cv2.imread(ldr_path).astype(np.float32)
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)

        tmqi = TMQI()
        score, s_score, n_score, _, _ = tmqi(hdr, ldr)
        print(score)
        avg_score += score
        num += 1

avg_score = avg_score/num
print('average score:{}'.format(avg_score))