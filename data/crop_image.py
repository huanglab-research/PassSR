import os
from PIL import Image
import argparse
import os
import shutil
import os
import random
from PIL import Image

def crop_dataset_to_fixed_size(dataset_path_hr,dataset_path_lr,target_size):
    # 获取数据集中的所有图像文件
    image_files_hr = [f for f in os.listdir(dataset_path_hr) if os.path.isfile(os.path.join(dataset_path_hr, f))]
    image_files_lr = [f for f in os.listdir(dataset_path_lr) if os.path.isfile(os.path.join(dataset_path_lr, f))]

    # 遍历每个图像文件
    for image_file_hr in image_files_hr:

        name_parts = image_file_hr.split('x2')
        name_parts.append(3)
        name_parts[2]=name_parts[1]
        name_parts[1]='x1'
        image_file_lr = ''.join(name_parts[0:3])
        if any(image_file_lr in file_lr for file_lr in image_files_lr):
            print(image_file_lr)
            image_path_hr = os.path.join(dataset_path_hr, image_file_hr)
            image_path_lr = os.path.join(dataset_path_lr, image_file_lr)

            # 打开图像
            image_hr = Image.open(image_path_hr)
            image_lr = Image.open(image_path_lr)
            # 获取图像的宽度和高度
            width, height = image_lr.size

            # 计算裁剪的起始坐标
            left = random.randint(0, width - target_size)
            top = random.randint(0, height - target_size)

            # 计算裁剪的结束坐标
            right = left + target_size
            bottom = top + target_size

            # 裁剪图像
            cropped_image_lr = image_lr.crop((left, top, right, bottom))
            cropped_image_hr = image_hr.crop((left*2,top*2,right*2,bottom*2))

            # 保存裁剪后的图像，覆盖原始图像
            cropped_image_hr.save('/home/haida/data/zuochenjuan/SR3_plus/dataset/DRealSR_128_256/hr/{}'.format(image_file_hr))
            cropped_image_lr.save('/home/haida/data/zuochenjuan/SR3_plus/dataset/DRealSR_128_256/lr/{}'.format(image_file_lr))

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-p', '--path', type=str)
    #parser.add_argument('-t','--target_size', type=int, default=64)

    #args = parser.parse_args()
    crop_dataset_to_fixed_size("/home/haida/data/zuochenjuan/SR3_plus/dataset/DRealSR_128_256/train_HR","/home/haida/data/zuochenjuan/SR3_plus/dataset/DRealSR_128_256/train_LR",128)

    '''
    # 定义文件夹路径
    folder_a = '/home/haida/data/zuochenjuan/SR3_plus/dataset/DRealSR_64_256/train_HR'
    folder_b = '/home/haida/data/zuochenjuan/SR3_plus/dataset/DRealSR_64_256/train_LR'
    folder_c = '/home/haida/data/zuochenjuan/SR3_plus/dataset/DRealSR_64_256/lr_64'

    # 获取文件夹a中的文件名列表
    files_a = os.listdir(folder_a)
    # 遍历文件夹b中的文件
    for filename in os.listdir(folder_b):
        
        # 提取文件名的关键部分（pa_x1_5_1）
        name_parts = filename.split('_x1_')
        name_parts.append(3)
        name_parts[2]=name_parts[1]
        name_parts[1]='x4'
        key_part = '_'.join(name_parts[0:3])
        #print(key_part)
        # 检查是否存在与文件夹a中对应的文件
        if any(key_part in file_a for file_a in files_a):
            #print(filename)
            # 构建文件的完整路径
            file_path_b = os.path.join(folder_b, filename)
            file_path_c = os.path.join(folder_c, filename)

            # 复制文件到文件夹c
            shutil.copy(file_path_b, file_path_c)    
        
    
    '''

    
