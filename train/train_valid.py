import time

import torch
import numpy as np
from tqdm import tqdm

from network.metrics import name2key_metrics
from train.train_tools import to_cuda
from utils.base_utils import load_cfg, to_cuda, color_map_backward, make_dir
from skimage.io import imsave

def save_renderings(output_dir, qi, render_info, h, w, save_fine_only=False):
    def output_image(suffix):
        if f'pixel_colors_{suffix}' in render_info:
            render_image = color_map_backward(render_info[f'pixel_colors_{suffix}'].cpu().numpy().reshape([h, w, 3]))
            imsave(f'{output_dir}/{qi}-{suffix}.jpg', render_image)

    if not save_fine_only:
        output_image('nr')
    output_image('nr_fine')

def save_depth(output_dir, qi, render_info, h, w, depth_range, gt_depth=None):
    suffix='fine'
    if f'render_depth_consistent_{suffix}' in render_info:
        depth = render_info[f'render_depth_consistent_{suffix}'].cpu().numpy().reshape([h, w])
        if gt_depth is not None:
            mask_depth = (gt_depth > 0)
            print('consistent depth quality', np.mean(np.abs(gt_depth[mask_depth] - depth[mask_depth])))
        near, far = depth_range
        depth = np.clip(depth, a_min=near, a_max=far)
        depth = (1/depth - 1/near)/(1/far - 1/near)
        depth = color_map_backward(depth)
        imsave(f'{output_dir}/{qi}-render_depth_consistent-{suffix}-depth.png', depth)
        

    if f'render_depth_{suffix}' in render_info:
        depth = render_info[f'render_depth_{suffix}'].cpu().numpy().reshape([h, w])
        if gt_depth is not None:
            print('nerf depth quality', np.mean(np.abs(gt_depth[mask_depth] - depth[mask_depth])))
        near, far = depth_range
        depth = np.clip(depth, a_min=near, a_max=far)
        depth = (1/depth - 1/near)/(1/far - 1/near)
        depth = color_map_backward(depth)
        imsave(f'{output_dir}/{qi}-render_depth{suffix}-depth.png', depth)

class ValidationEvaluator:
    def __init__(self,cfg):
        self.key_metric_name=cfg['key_metric_name']
        self.key_metric=name2key_metrics[self.key_metric_name]

    def __call__(self, model, losses, eval_dataset, step, model_name, val_set_name=None, save_dir=None):
        if val_set_name is not None: model_name=f'{model_name}-{val_set_name}'
        model.eval()
        eval_results={}
        begin=time.time()
        for data_i, data in tqdm(enumerate(eval_dataset)):
            data = to_cuda(data)
            data['eval']=True
            data['step']=step
            with torch.no_grad():
                outputs = model(data)
                for loss in losses:
                    loss_results=loss(outputs, data, step, data_index=data_i, model_name=model_name)
                    for k,v in loss_results.items():
                        if k in ['loss_prompt', 'loss_consistent_prompt', 'loss_smooth_prompt'] :
                            continue
                        if type(v)==torch.Tensor:
                            v=v.detach().cpu().numpy()

                        if k in eval_results:
                            eval_results[k].append(v)
                        else:
                            eval_results[k]=[v]

                if save_dir:
                    import pickle
                    _, _, h, w = outputs['que_imgs_info']['imgs'].shape

                    consistent_weights = outputs['consistent_weights'].cpu().detach().numpy()[0].reshape(512,640,64)
                    que_depth = outputs['que_depth'].cpu().detach().numpy()[0].reshape(512,640,64)
                    rgb_feat_sum = outputs['rgb_feat_sum'].cpu().detach().numpy()[0].reshape(512,640,64, 7, 7)
                    pk_dict = {'consistent_weights': consistent_weights,
                        'que_depth': que_depth,
                        'rgb_feat_sum': rgb_feat_sum
                    }

                    with open(f'{save_dir}/consis_weight_cosin.pkl', 'wb') as handle:
                        pickle.dump(pk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # print('consistent_weights', consistent_weights.shape)
                    # save_path = f'{save_dir}/consis_weight.np'
                    # np.save(save_path, consistent_weights)
                    quit()

                    save_renderings(save_dir, data_i, outputs, h, w, save_fine_only=True)

                    # breakpoint()
                    gt_depth = outputs['que_imgs_info']['depth'].cpu().numpy()[0,0]
                    que_depth_ranges = outputs['que_imgs_info']['depth_range'][0].cpu().detach().numpy()
                    save_depth(save_dir, data_i, outputs, h, w, que_depth_ranges, gt_depth=gt_depth)

        for k,v in eval_results.items():
            if k == 'loss_prompt':
                continue
            eval_results[k]=np.concatenate(v,axis=0)

        key_metric_val=self.key_metric(eval_results)
        eval_results[self.key_metric_name]=key_metric_val
        print('eval cost {} s'.format(time.time()-begin))
        return eval_results, key_metric_val
