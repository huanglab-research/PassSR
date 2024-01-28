import os
from PIL import Image
import argparse
import os
import shutil
import os
import random
from PIL import Image

def crop_dataset_to_fixed_size(dataset_path_hr,dataset_path_lr,target_size):

    image_files_hr = [f for f in os.listdir(dataset_path_hr) if os.path.isfile(os.path.join(dataset_path_hr, f))]
    image_files_lr = [f for f in os.listdir(dataset_path_lr) if os.path.isfile(os.path.join(dataset_path_lr, f))]


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


            image_hr = Image.open(image_path_hr)
            image_lr = Image.open(image_path_lr)

            width, height = image_lr.size


            left = random.randint(0, width - target_size)
            top = random.randint(0, height - target_size)

            right = left + target_size
            bottom = top + target_size


            cropped_image_lr = image_lr.crop((left, top, right, bottom))
            cropped_image_hr = image_hr.crop((left*2,top*2,right*2,bottom*2))


            cropped_image_hr.save('/path/dataset/DRealSR_128_256/hr/{}'.format(image_file_hr))
            cropped_image_lr.save('/path/dataset/DRealSR_128_256/lr/{}'.format(image_file_lr))

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-p', '--path', type=str)
    #parser.add_argument('-t','--target_size', type=int, default=64)

    #args = parser.parse_args()
    crop_dataset_to_fixed_size("/path/dataset/DRealSR_128_256/train_HR","/path/dataset/DRealSR_128_256/train_LR",128)
    
