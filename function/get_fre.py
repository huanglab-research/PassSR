import os
import cv2
import numpy as np
import depart_freq as DeFreq
def depart(path_sr,path_sr0):

    path_save="/path/dataset/DIV2K_val_64_256/sr_fre/"
    # 遍历数据集A中的图像
    for filename in os.listdir(path_sr):
        print(filename)
        # 构建图像文件路径
        image_a_path = os.path.join(path_sr, filename)
        image_b_path = os.path.join(path_sr0, filename)
        image_c_path = os.path.join(path_save, filename)

        # 读取图像
        image_a = cv2.imread(image_a_path)
        image_b = cv2.imread(image_b_path)

        freq_a=DeFreq.depart_frequence(image_a)
        freq_b=DeFreq.depart_frequence(image_b)
        
        diff_freq=np.array(freq_a)-np.array(freq_b)

        # 保存差值图像到数据集C
        cv2.imwrite(image_c_path, np.transpose(diff_freq, (1, 2, 0)))


depart("/path/dataset/DIV2K_val_64_256/sr_64_256","/path/dataset/DIV2K_val_64_256/sr0_64_256")