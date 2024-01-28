"""
Author  : Xu fuyong
Time    : created by 2019/7/17 17:41

"""
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from model import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default="/path/CNN_SR/outputs/x_Dr_deep2/epoch_150.pth")
    parser.add_argument('--image-file', type=str, default="/path/dataset/RealSR_128_256/lr/")
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    cudnn.benchmark = True
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    
    image_files = os.listdir(args.image_file)
    sum_psnr=0
    i=0
    for file_name in image_files:
        i+=1
        image_path="".join([args.image_file,file_name])
        
        image = pil_image.open(image_path).convert('RGB')
        
        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        #image = image.resize((64, 64), resample=pil_image.BICUBIC)
        image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
        image_path_save_b="".join(["/path/dataset/RealSR_128_256/sr/",file_name])
        image.save(image_path_save_b)
        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        psnr = calc_psnr(y, preds)
        sum_psnr+=psnr
        print('PSNR: {:.2f}'.format(psnr))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        image_path_save="".join(["/path/dataset/RealSR_128_256/cnnsr/",file_name])
        output.save(image_path_save)

    print("avl_psnr={}".format(1.0*sum_psnr/i))
