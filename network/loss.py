import torch
import torch.nn as nn

from network.ops import interpolate_feats


class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys=keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass

class ConsistencyLoss(Loss):
    default_cfg={
        'use_ray_mask': False,
        'use_dr_loss': False,
        'use_dr_fine_loss': False,
        'use_nr_fine_loss': False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_prob','loss_prob_fine'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'hit_prob_self' not in data_pr: return {}
        prob0 = data_pr['hit_prob_nr'].detach()     # qn,rn,dn
        prob1 = data_pr['hit_prob_self']            # qn,rn,dn
        if self.cfg['use_ray_mask']:
            ray_mask = data_pr['ray_mask'].float()  # 1,rn
        else:
            ray_mask = 1
        ce = - prob0 * torch.log(prob1 + 1e-5) - (1 - prob0) * torch.log(1 - prob1 + 1e-5)
        outputs={'loss_prob': torch.mean(torch.mean(ce,-1),1)}
        if 'hit_prob_nr_fine' in data_pr:
            prob0 = data_pr['hit_prob_nr_fine'].detach()     # qn,rn,dn
            prob1 = data_pr['hit_prob_self_fine']            # qn,rn,dn
            ce = - prob0 * torch.log(prob1 + 1e-5) - (1 - prob0) * torch.log(1 - prob1 + 1e-5)
            outputs['loss_prob_fine']=torch.mean(torch.mean(ce,-1),1)
        return outputs

class RenderLoss(Loss):
    default_cfg={
        'use_ray_mask': True,
        'use_dr_loss': False,
        'use_dr_fine_loss': False,
        'use_nr_fine_loss': False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_rgb'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        rgb_gt = data_pr['pixel_colors_gt'] # 1,rn,3
        rgb_nr = data_pr['pixel_colors_nr'] # 1,rn,3
        def compute_loss(rgb_pr, rgb_gt, fine=False):
            loss=torch.sum((rgb_pr-rgb_gt)**2,-1)        # b,n

            if self.cfg['use_ray_mask']:
                # ray_mask = data_pr['ray_mask_fine'].float() if fine else data_pr['ray_mask'].float()# 1,rn
                ray_mask = data_pr['ray_mask'].float()
                loss = torch.sum(loss*ray_mask,1)/(torch.sum(ray_mask,1)+1e-3)
            else:
                loss = torch.mean(loss, 1)
            return loss

        results = {'loss_rgb_nr': compute_loss(rgb_nr, rgb_gt)}
        if self.cfg['use_dr_loss']:
            rgb_dr = data_pr['pixel_colors_dr']  # 1,rn,3
            results['loss_rgb_dr'] = compute_loss(rgb_dr, rgb_gt)
        if self.cfg['use_dr_fine_loss']:
            results['loss_rgb_dr_fine'] = compute_loss(data_pr['pixel_colors_dr_fine'], rgb_gt)
        if self.cfg['use_nr_fine_loss']:
            results['loss_rgb_nr_fine'] = compute_loss(data_pr['pixel_colors_nr_fine'], rgb_gt, fine=True)
        return results

class DepthLoss(Loss):
    default_cfg={
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
    }
    def __init__(self, cfg):
        super().__init__(['loss_depth'])
        self.cfg={**self.default_cfg,**cfg}
        if self.cfg['depth_loss_type']=='smooth_l1':
            self.loss_op=nn.SmoothL1Loss(reduction='none',beta=self.cfg['depth_loss_l1_beta'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'true_depth' not in data_gt['ref_imgs_info']:
            return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr'].device)}
        coords = data_pr['depth_coords'] # rfn,pn,2
        depth_pr = data_pr['depth_mean'] # rfn,pn
        depth_maps = data_gt['ref_imgs_info']['true_depth'] # rfn,1,h,w
        rfn, _, h, w = depth_maps.shape
        depth_gt = interpolate_feats(
            depth_maps,coords,h,w,padding_mode='border',align_corners=True)[...,0]   # rfn,pn

        # transform to inverse depth coordinate
        depth_range = data_gt['ref_imgs_info']['depth_range'] # rfn,2
        near, far = -1/depth_range[:,0:1], -1/depth_range[:,1:2] # rfn,1
        def process(depth):
            depth = torch.clamp(depth, min=1e-5)
            depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth
        depth_gt = process(depth_gt)

        # compute loss
        def compute_loss(depth_pr):
            if self.cfg['depth_loss_type']=='l2':
                loss = (depth_gt - depth_pr)**2
            elif self.cfg['depth_loss_type']=='smooth_l1':
                loss = self.loss_op(depth_gt, depth_pr)

            if data_gt['scene_name'].startswith('gso'):
                depth_maps_noise = data_gt['ref_imgs_info']['depth']  # rfn,1,h,w
                depth_aug = interpolate_feats(depth_maps_noise, coords, h, w, padding_mode='border', align_corners=True)[..., 0]  # rfn,pn
                depth_aug = process(depth_aug)
                mask = (torch.abs(depth_aug-depth_gt)<self.cfg['depth_correct_thresh']).float()
                loss = torch.sum(loss * mask, 1) / (torch.sum(mask, 1) + 1e-4)
            else:
                loss = torch.mean(loss, 1)
            return loss

        outputs = {'loss_depth': compute_loss(depth_pr)}
        if 'depth_mean_fine' in data_pr:
            outputs['loss_depth_fine'] = compute_loss(data_pr['depth_mean_fine'])
        return outputs


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class VirtualRenderLoss(Loss):
    default_cfg={
        'use_ray_mask': True,
        'use_nr_fine_loss': False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_rgb_virtual'])
        self.ssim = SSIM().cuda()

    def __call__(self, data_pr, data_gt, step, **kwargs):
        rgb_gt = data_pr['virtual_pixel_colors_gt'] # 1,rn,3, rfn
        rgb_nr = data_pr['virtual_pixel_colors_nr'] # 1,rn,3

        

        def compute_loss_virtual(rgb_pr, rgb_gt, fine=False):

            rgb_gt = rgb_gt.reshape(1, 32, 32, 3, -1).permute(4, 0, 3, 1, 2) # rfn, 1, c, h ,w
            rgb_pr = rgb_pr.reshape(1, 32, 32, 3).permute(0, 3, 1, 2) # 1, c, h, w

            losses = []
            for r in range(rgb_gt.shape[0]):
                rgb_gt_single = rgb_gt[r,...]
                abs_diff = torch.abs(rgb_gt_single - rgb_pr)
                l1_loss = abs_diff.mean(1)

                ssim_loss = self.ssim(rgb_pr, rgb_gt_single).mean(1)
                loss = 0.85 * ssim_loss + 0.15 * l1_loss # 1, h, w
                losses.append(loss.reshape(loss.shape[0], -1))
            # loss=torch.sum((rgb_pr-rgb_gt)**2,-1)
            # l2_loss = torch.sum((rgb_pr.unsqueeze(-1) - rgb_gt)**2, 2)  # 1, rn, rfn
            losses = torch.stack(losses, dim=0) # rfn, 1, rn

            temporal = 10.
            final_loss = torch.logsumexp(losses * temporal, dim=0) / temporal # 1, rn
            # loss = torch.log(torch.sum(torch.exp(loss * temporal), dim=-1)) / temporal
            # loss = torch.sum((rgb_pr.unsqueeze(-1) - rgb_gt)**2, 2) # 1, rn, rfn
            # loss = torch.min(loss, -1)[0]
            if self.cfg['use_ray_mask']:
                # ray_mask = data_pr['virtual_pixel_mask_fine'].float() if fine else data_pr['virtual_pixel_mask'].float()# 1,rn
                ray_mask = data_pr['virtual_pixel_mask'].float()
                final_loss = torch.sum(final_loss * ray_mask,1) / (torch.sum(ray_mask, 1) + 1e-3)
            else:
                final_loss = torch.mean(final_loss, 1)

            # print('final_loss', final_loss)
            return final_loss * 0.1

        results = {'loss_rgb_nr_virtual': compute_loss_virtual(rgb_nr, rgb_gt)}
        if self.cfg['use_nr_fine_loss']:
            results['loss_rgb_nr_fine_virtual'] = compute_loss_virtual(data_pr['virtual_pixel_colors_nr_fine'], rgb_gt, fine=True)
        
        # print('debug', results['loss_rgb_nr_virtual'].shape)
        # if self.cfg['use_nr_fine_loss']:
        #     results['loss_rgb_nr_fine'] = compute_loss_virtual(data_pr['pixel_colors_nr_fine'], rgb_gt, fine=True)
        return results

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

class VirtualSmoothDepth(Loss):
    default_cfg={
        'use_ray_mask': True,
        'use_nr_fine_loss': False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_virtual_smooth_depth'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        def compute_smooth_depth_virtual(depth, rgb):

            depth = depth.reshape(1, 32, 32, 1).permute(0, 3, 1, 2) # 1, 1, h, w
            rgb = rgb.reshape(1, 32, 32, 3).permute(0, 3, 1, 2) # 1, c, h, w

            mean_depth = depth.mean(2, True).mean(3, True)
            norm_depth = depth / (mean_depth + 1e-7)

            smooth_loss = get_smooth_loss(norm_depth, rgb.detach())

            return torch.mean(smooth_loss)

        depth_virtual = data_pr['depth_virtual'] # 1,rn,
        rgb_pr = data_pr['virtual_pixel_colors_nr'] # 1,rn,3

        depth_virtual_fine = data_pr['depth_virtual_fine'] # 1,rn,
        rgb_pr_fine = data_pr['virtual_pixel_colors_nr_fine'] # 1,rn,3

        results = {'loss_virtual_smooth_depth': compute_smooth_depth_virtual(depth_virtual, rgb_pr),
                    'loss_virtual_smooth_depth_fine': compute_smooth_depth_virtual(depth_virtual_fine, rgb_pr_fine)}

        return results

name2loss={
    'render': RenderLoss,
    'virtual_render': VirtualRenderLoss,
    'depth': DepthLoss,
    'consist': ConsistencyLoss,
    'virtual_smooth_depth': VirtualSmoothDepth,
}