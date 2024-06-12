"""
Author  : Anonymous
Time    : created by 2024-1-28
"""
import argparse
import cv2
import numpy as np
import pywt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from scipy import ndimage
def depart_merge(image):
    image= cv2.imread(image+'.png')
    kernel_sharpening = np.array([[-1.5, -1, -1.5], [-1, 11, -1], [-1.5, -1, -1.5]])
    adjusted_image2_shar = cv2.filter2D(adjusted_image2, -1, kernel_sharpening)
    kernel_smoothing = np.ones((2, 2)) / 4
    adjusted_image2_shar = cv2.filter2D(adjusted_image2_shar, -1, kernel_smoothing)
    #adjusted_image = cv2.GaussianBlur(adjusted_image, (5, 5), 0)
    adjusted_image2 = cv2.Laplacian(adjusted_image2_shar, cv2.CV_8U)
    adjusted_image2 = cv2.GaussianBlur(adjusted_image2_shar, (3, 3), 0)
    adjusted_image = cv2.addWeighted(adjusted_image2, 1, adjusted_image1, 0, 0)
    
    #adjusted_image=cv2.bilateralFilter(adjusted_image, 40, 15, 22)
    saturation_factor = 1.03
    hsv_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    #adjusted_image = cv2.medianBlur(adjusted_image, 3)  
    cv2.imwrite("/path/function/results_sr/0_enhance_inf_0{}_LR4.png".format(i),adjusted_image)
    return adjusted_image
def depart_frequence(image,t):
    #image = cv2.imread(image+'.png')#(512, 512, 3)
    image = image.cpu().numpy() 
    image = np.transpose(image, (1, 2, 0))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2

    low_cutoff = 10  
    high_cutoff = 20  
    lowpass = np.zeros((rows, cols), np.float32)
    lowpass[crow - low_cutoff:crow + low_cutoff, ccol - low_cutoff:ccol + low_cutoff] = 1
    highpass = np.ones((rows, cols), np.float32)
    highpass[crow - high_cutoff:crow + high_cutoff, ccol - high_cutoff:ccol + high_cutoff] = 0
    low_frequency = fshift * lowpass
    low_image = np.fft.ifftshift(low_frequency)
    low_image = np.fft.ifft2(low_image)
    low_image = np.abs(low_image)

    '''
    high_frequency = fshift * highpass
    high_image = np.fft.ifftshift(high_frequency)
    high_image = np.fft.ifft2(high_image)
    high_image = np.abs(high_image)    
    '''
    #blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    blurred_image = gray_image.astype(np.float32)
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_32F)
    high_image = np.abs(laplacian)
    mid_frequency = fshift * (1 - lowpass - highpass)
    mid_image = np.fft.ifftshift(mid_frequency)
    mid_image = np.fft.ifft2(mid_image)
    mid_image = np.abs(mid_image)
    cv2.imwrite('/path/save/high_frequency.png', high_image)
    cv2.imwrite('/path/save/low_frequency.png', low_image)
    cv2.imwrite('/path/save/mid_frequency.png', mid_image)
    return low_image,mid_image,high_image
