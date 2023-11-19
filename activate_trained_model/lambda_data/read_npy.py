import cv2
import numpy as np

path = 'ldr_avg_hist_900_images_20_bins.npy'
data = np.load(path, allow_pickle=True)
print(data)