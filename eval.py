import argparse
import os
import lpips

import torch
from skimage.io import imread
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity
from utils.base_utils import color_map_forward
from PIL import Image
from torchvision import transforms as T
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
transform = T.ToTensor()
mse2psnr = lambda x : -10. * np.log(x) / np.log(10.)

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    # print(image_pred.shape, image_gt.shape)
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return np.mean(value)
    return value

def psnr_new(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*np.log10(mse(image_pred, image_gt, valid_mask, reduction))

class Evaluator:
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='vgg').cuda().eval()
        # self.loss_fn_alex = lpips.LPIPS(net='alex').cuda().eval()

    def eval_metrics_img(self,gt_img, pr_img):
        gt_img = color_map_forward(gt_img)
        pr_img = color_map_forward(pr_img)
        # psnr = tf.image.psnr(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        # ssim = tf.image.ssim(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        mask = np.any(gt_img > 0, axis=-1)
        # psnr = mse2psnr(np.mean((pr_img[mask] - gt_img[mask])**2))
        psnr = psnr_new(pr_img, gt_img)
        ssim = structural_similarity(gt_img, pr_img, multichannel=True)
        with torch.no_grad():
            gt_img_th = torch.from_numpy(gt_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            pr_img_th = torch.from_numpy(pr_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            score = float(self.loss_fn_alex(gt_img_th, pr_img_th).flatten()[0].cpu().numpy())
        return float(psnr), float(ssim), score


    def eval(self, dir_gt, dir_pr):
        results=[]
        num = len(os.listdir(dir_gt))
        for k in tqdm(range(0, num)):
            pr_img = imread(f'{dir_pr}/{k}-nr_fine.jpg')
            gt_img = imread(f'{dir_gt}/{k}.jpg')

            img = np.array(Image.open(f'{dir_gt}/{k}.jpg'))
            print('shape', img.shape, gt_img.shape)
            # img = transform(img).view(3,-1).permute(1,0).numpy()

            # print(img - gt_img.reshape(-1,3).astype(np.float32)/255.0)

            psnr, ssim, lpips_score = self.eval_metrics_img(gt_img, pr_img)
            results.append([psnr,ssim,lpips_score])
        psnr, ssim, lpips_score = np.mean(np.asarray(results),0)

        msg=f'psnr {psnr:.4f} ssim {ssim:.4f} lpips {lpips_score:.4f}'
        print(msg)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_gt', type=str, default='data/render/fern/gt')
    parser.add_argument('--dir_pr', type=str, default='data/render/fern/neuray_gen_depth-pretrain-eval')
    flags = parser.parse_args()
    evaluator = Evaluator()
    evaluator.eval(flags.dir_gt, flags.dir_pr)
