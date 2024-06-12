"""
Author  : Anonymous
Time    : created by 2024-1-28
"""
import cv2
import numpy as np
def dct_frequence(image):
    #image = cv2.imread(image+'.png')#(512, 512, 3)
    image = image.cpu().numpy() 
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(image))
    threshold = 0.01  
    mask = np.ones_like(dct)
    mask[dct < threshold * np.max(dct)] = 0
    idct = cv2.idct(mask * dct)
    cv2.imwrite("/path/func/mask.png", idct.astype(np.uint8))
    mean_value = np.mean(idct)
    binary_spectrum = np.where(idct > 1.5*mean_value, 0, 255)
    #print(binary_spectrum)
    cv2.imwrite("/path/func/mask_01.png", binary_spectrum)
    return binary_spectrum
