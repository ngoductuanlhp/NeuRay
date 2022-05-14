import torch
import torch.nn as nn

from network.ops import interpolate_feats
import numpy as np

class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys=keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass

class PromptLoss(Loss):
    default_cfg={}
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_prompt'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'out_prompt' not in data_pr: 
            return {'loss_prompt': torch.Tensor([0.])}

        m_hit_prob = data_pr['m_hit_prob']
        m_hit_prob_mask = (m_hit_prob >= 0.2).detach()
        if torch.count_nonzero(m_hit_prob_mask) == 0:
            return {'loss_prompt': torch.Tensor([0.])}
        out_ibr = data_pr['out_ibr'][m_hit_prob_mask.repeat(1,1,1,data_pr['out_ibr'].shape[-1])].detach()
        out_prompt = data_pr['out_prompt'][m_hit_prob_mask.repeat(1,1,1,data_pr['out_prompt'].shape[-1])]

        # print('debug shape', data_pr['out_ibr'].shape, data_pr['out_prompt'].shape)
        loss_prompt = torch.mean((out_prompt - out_ibr)**2) * 1e-4
        outputs = {'loss_prompt': loss_prompt}
        return outputs

class ConsistentPromptLoss(Loss):
    default_cfg={}
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_consistent_prompt'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'consistent_weights' not in data_pr: 
            return {'loss_consistent_prompt': torch.Tensor([0.])}

        consistent_weights = data_pr['consistent_weights']
        alpha_values = data_pr['alpha_values']

        mask = (consistent_weights > 0.5)
        loss = torch.mean( consistent_weights[mask] * (1 - alpha_values[mask])) * 1e-2
        outputs = {'loss_consistent_prompt': loss}
        return outputs

class SmoothPromptLoss(Loss):
    default_cfg={}
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_smooth_prompt'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'prompt_feats' not in data_pr: 
            return {'loss_smooth_prompt': torch.Tensor([0.])}

        prompt_feats = data_pr['prompt_feats']

        loss = 0
        for k in prompt_feats.keys():
            # loss += self.compute_loss(feat_line)*1e-3
            loss += torch.mean((prompt_feats[k][:,:,1:,:] - prompt_feats[k][:,:,:-1,:])**2)*1e-3
        outputs = {'loss_smooth_prompt': loss}
        return outputs

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
        def compute_loss(rgb_pr,rgb_gt):
            loss=torch.sum((rgb_pr-rgb_gt)**2,-1)        # b,n
            if self.cfg['use_ray_mask']:
                ray_mask = data_pr['ray_mask'].float() # 1,rn
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
            results['loss_rgb_nr_fine'] = compute_loss(data_pr['pixel_colors_nr_fine'], rgb_gt)
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

class TVFullPromptLoss(Loss):
    default_cfg={}
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_tv_full_prompt'])
        self.percentage_sample = 0.4
    
    def sampling_indices(self, Dx,Dy,Dz):
        '''
        Subsample indices for prompt tensor
        '''
        xrange = torch.arange(Dx - 1)
        yrange = torch.arange(Dy - 1)
        zrange = torch.arange(Dz - 1)
        xx, yy, zz = torch.meshgrid(xrange,yrange,zrange)
        grid = torch.stack([xx.flatten(),yy.flatten(),zz.flatten()],dim=-1)
        sampling_indices = torch.from_numpy(np.random.choice(len(grid),int(self.percentage_sample * len(grid)),replace=False))
        selected_indices = grid[sampling_indices]
        return selected_indices

    def __call__(self, data_pr, data_gt, step, **kwargs):
        prompt = data_pr['prompt_feats']
        subsample_indices = self.sampling_indices(prompt.shape[2], prompt.shape[3], prompt.shape[4])
        prompt = prompt.permute(0,2,3,4,1)
        prompt_feats = prompt[0,subsample_indices[0],subsample_indices[1],subsample_indices[2],:]
        prompt_featsx = prompt[0,subsample_indices[0]+1,subsample_indices[1],subsample_indices[2],:]
        prompt_featsy = prompt[0,subsample_indices[0],subsample_indices[1]+1,subsample_indices[2],:]
        prompt_featsz = prompt[0,subsample_indices[0],subsample_indices[1],subsample_indices[2]+1,:]
        loss = torch.sqrt(
            torch.norm(prompt_featsx - prompt_feats,dim=-1,p=2) + \
            torch.norm(prompt_featsy - prompt_feats,dim=-1,p=2) + \
            torch.norm(prompt_featsz - prompt_feats,dim=-1,p=2)
        )

        outputs = {
            'loss_tv_full_prompt': torch.mean(loss) * 1e-1
        }

        return outputs



name2loss={
    'render': RenderLoss,
    'depth': DepthLoss,
    'consist': ConsistencyLoss,
    'prompt': PromptLoss,
    'smooth_prompt': SmoothPromptLoss,
    'consistent_prompt': ConsistentPromptLoss,
    'tv_full_prompt': TVFullPromptLoss
}