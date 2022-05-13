import torch
import torch.nn as nn
from easydict import EasyDict

from network.ibrnet import IBRNetWithNeuRay


def get_dir_diff(prj_dir,que_dir):
    rfn, qn, rn, dn, _ = prj_dir.shape
    dir_diff = prj_dir - que_dir.unsqueeze(0)  # rfn,qn,rn,dn,3
    dir_dot = torch.sum(prj_dir * que_dir.unsqueeze(0), -1, keepdim=True)
    dir_diff = torch.cat([dir_diff, dir_dot], -1)  # rfn,qn,rn,dn,4
    dir_diff = dir_diff.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)
    return dir_diff

class DefaultAggregationNet(nn.Module):
    default_cfg={
        'sample_num': 64,
        'neuray_dim': 32,
        'use_img_feats': False,
    }
    def __init__(self,cfg):
        super().__init__()
        self.cfg={**self.default_cfg, **cfg}
        # args = EasyDict(anti_alias_pooling=False,local_rank=0)
        dim = self.cfg['neuray_dim']
        self.agg_impl = IBRNetWithNeuRay(dim,n_samples=self.cfg['sample_num'])
        self.prob_embed = nn.Sequential(
            nn.Linear(2+32, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, prj_dict, que_dir, prompt):
        """
        :param prj_dict
             prj_ray_feats: rfn,qn,rn,dn,f
             prj_hit_prob:  rfn,qn,rn,dn,1
             prj_vis:       rfn,qn,rn,dn,1
             prj_alpha:     rfn,qn,rn,dn,1
             prj_rgb:       rfn,qn,rn,dn,3
             prj_dir:       rfn,qn,rn,dn,3
        :param que_dir:       qn,rn,dn,3
        :return: qn,rn,dn
        """
        hit_prob_val = (prj_dict['hit_prob']-0.5)*2
        vis_val = (prj_dict['vis']-0.5)*2

        prj_hit_prob, prj_vis, prj_rgb, prj_dir, prj_ray_feats = \
            hit_prob_val, vis_val, prj_dict['rgb'], prj_dict['dir'], prj_dict['ray_feats']
        rfn,qn,rn,dn,_ = hit_prob_val.shape

        
        # db_hit_prob = prj_dict['hit_prob'].flatten()
        # print('db_hit_prob range', torch.min(db_hit_prob), torch.max(db_hit_prob), torch.mean(db_hit_prob))

        prob_embedding = self.prob_embed(torch.cat([prj_ray_feats, prj_hit_prob, prj_vis],-1))

        dir_diff = get_dir_diff(prj_dir, que_dir)
        valid_mask = prj_dict['mask']
        valid_mask = valid_mask.float() # rfn,qn,rn,dn
        valid_mask = valid_mask.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)

        prj_img_feats = prj_dict['img_feats']
        prj_img_feats = torch.cat([prj_rgb, prj_img_feats], -1)
        prj_img_feats = prj_img_feats.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3) # n_ray, n_sample, n_view, channel
        prob_embedding = prob_embedding.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)


        mean_hit_prob = torch.mean(prj_dict['hit_prob'].reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3), dim=2) # n_ray, n_sample, channel

        # FIXME hardcode
        # mean_hit_prob = torch.exp(-(1 - mean_hit_prob))

        outs, gt_ibr, consistent_weights = self.agg_impl(prj_img_feats, prob_embedding, dir_diff, valid_mask, prompt, mean_hit_prob)

        
            # # print('mean_hit_prob', mean_hit_prob.shape)
            # # colors_mix = mean_hit_prob * outs['out_ibr'][...,:3] + (1. - mean_hit_prob) * outs['out_prompt'][...,:3]
            # # density_mix = mean_hit_prob * outs['out_ibr'][...,3:4] + (1. - mean_hit_prob) * outs['out_prompt'][...,3:4]
            # colors_mix = outs['out_ibr'][...,:3]
            # density_mix = outs['out_ibr'][...,3:4]

            # return density_mix.reshape(qn,rn,dn), colors_mix.reshape(qn,rn,dn,3), mean_hit_prob.reshape(qn,rn,dn,1), outs['out_ibr'].reshape(qn,rn,dn,4), outs['out_prompt'].reshape(qn,rn,dn,4)
        colors = outs[...,:3] # qn*rn,dn,3
        density = outs[...,3] # qn*rn,dn,0
        return density.reshape(qn,rn,dn), colors.reshape(qn,rn,dn,3), mean_hit_prob, gt_ibr, consistent_weights.reshape(qn,rn,dn)


name2agg_net={
    'default': DefaultAggregationNet
}