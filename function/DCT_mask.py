import cv2
import numpy as np
def dct_frequence(image):
    #image = cv2.imread(image+'.png')#(512, 512, 3)
    image = image.cpu().numpy() 
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行DCT变换
    dct = cv2.dct(np.float32(image))

    # 生成频域掩码
    threshold = 0.01  # 阈值，用于选择保留哪些频率成分
    mask = np.ones_like(dct)
    mask[dct < threshold * np.max(dct)] = 0

    # 逆DCT变换
    idct = cv2.idct(mask * dct)

    # 显示原始图像和生成的图像
    cv2.imwrite("/path/func/mask.png", idct.astype(np.uint8))
    # 计算灰度频域图像的平均值
    mean_value = np.mean(idct)

    #高频为255
    binary_spectrum = np.where(idct > 1.5*mean_value, 0, 255)
    #print(binary_spectrum)
    cv2.imwrite("/path/func/mask_01.png", binary_spectrum)
    return binary_spectrum