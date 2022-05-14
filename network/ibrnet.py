import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn

# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var

class IBRNet(nn.Module):
    def __init__(self, in_feat_ch=32, n_samples=64, **kwargs):
        super(IBRNet, self).__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, mask):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out

class IBRNetWithNeuRay(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )

        # self.rgb_final_fc = nn.Sequential(
        #     nn.Linear(in_feat_ch+3, 32,),
        #     activation_func,
        #     nn.Linear(32, 3),
        #     nn.Sigmoid()
        # )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)
        # self.rgb_final_fc.apply(weights_init)

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, prompt, mean_hit_prob):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samplesblending_weights_valid, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]


        # rgb_feat: [n_rays, n_samples, n_views, n_feat] 
        rgb_feat1 = rgb_feat[:, :, :, None, :].repeat(1,1,1,num_views,1)  # [n_rays, n_samples, n_views, n_views, n_feat]
        rgb_feat2 = rgb_feat[:, :, None, :, :].repeat(1,1,num_views,1,1)  # [n_rays, n_samples, n_views, n_views, n_feat]

        # rgb_feat_mat = rgb_feat1 *rgb_feat2
        rgb_feat_sum = torch.sum(rgb_feat1 * rgb_feat2, dim =-1)
        # rgb_feat_sum = F.cosine_similarity(rgb_feat1, rgb_feat2, dim=-1)
        mask_rgb_feat_sum = torch.eye(num_views)[None, None, ...].type(torch.bool).to(rgb_feat_sum.device).repeat(rgb_feat_sum.shape[0], rgb_feat_sum.shape[1], 1, 1)
        rgb_feat_sum[mask_rgb_feat_sum] = -1e8
        rgb_feat_exp = torch.exp(rgb_feat_sum) # [n_rays, n_samples, n_views, n_views]
        

        # rgb_feat_exp_mean_row = torch.mean(rgb_feat_exp, dim=3) # [n_rays, n_samples, n_views]
        # rgb_feat_exp_mean_total = torch.mean(rgb_feat_exp_mean_row, dim=2)
        rgb_feat_exp_sum_row = torch.sum(rgb_feat_exp, dim=3)
        rgb_feat_exp_sum_total = torch.sum(rgb_feat_exp_sum_row, dim=2)
        rgb_feat_exp_sum_total = F.tanh(rgb_feat_exp_sum_total) # 0->1

        assert torch.all(rgb_feat_exp_sum_total >=0) and torch.all(rgb_feat_exp_sum_total <= 1)

        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat

        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
        
        # prompt feats
        # prompt_rgb_feats = prompt[..., :35].squeeze(0) # [n_rays, n_samples, 3]
        # prompt_sigma_feats = prompt[..., 35:].squeeze(0) # [n_rays, n_samples, 16]
        prompt_sigma_feats = prompt.squeeze(0)

        # neuray layer 0
        weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x) # [n_rays, n_samples, n_views, 32]

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]

        globalfeat_clone = globalfeat.clone().detach()

        # NOTE combine weights
        globalfeat = rgb_feat_exp_sum_total.unsqueeze(-1) * globalfeat + (1. - rgb_feat_exp_sum_total.unsqueeze(-1)) * prompt_sigma_feats
        # globalfeat = mean_hit_prob * globalfeat + (1. - mean_hit_prob) * prompt_sigma_feats
        # globalfeat = prompt_sigma_feats

        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding

        # print('ibr', globalfeat.shape, self.pos_encoding.shape)
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out1 = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # NOTE mask low consistent point by 0
        sigma_out1 = sigma_out1.masked_fill(rgb_feat_exp_sum_total[..., None] < 0.2, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending

        rgb_out1 = torch.sum(rgb_in*blending_weights_valid, dim=2)
        # rgb_feat_blend = torch.sum(rgb_feat * blending_weights_valid, dim=2)
        # NOTE combine
        # rgb_feat_blend = mean_hit_prob * rgb_feat_blend + (1. - mean_hit_prob) * prompt_rgb_feats


        # rgb_out1 = self.rgb_final_fc(rgb_feat_blend)
        # rgb_feat_blend_clone = rgb_feat_blend.clone().detach()
        # gt_ibr = torch.cat([rgb_feat_blend_clone, globalfeat_clone], axis=-1)
        gt_ibr = globalfeat_clone

        # out = torch.cat([rgb_out, sigma_out1], dim=-1)

        # return out
        out1 = torch.cat([rgb_out1, sigma_out1], dim=-1)
        return out1, gt_ibr

class IBRNetWithNeuRay2(nn.Module):
    def __init__(self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, **kwargs):
        super().__init__()
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        # self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5+neuray_in_dim, 64),
        #                              activation_func,
        #                              nn.Linear(64, 32),
        #                              activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*5, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.neuray_fc = nn.Sequential(
            nn.Linear(neuray_in_dim, 8,),
            activation_func,
            nn.Linear(8, 1),
        )

        # self.rgb_final_fc = nn.Sequential(
        #     nn.Linear(in_feat_ch+3, 32,),
        #     activation_func,
        #     nn.Linear(32, 3),
        #     nn.Sigmoid()
        # )

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)
        # self.rgb_final_fc.apply(weights_init)

    def change_pos_encoding(self,n_samples):
        self.pos_encoding = self.posenc(16, n_samples=n_samples)

    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(0)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, neuray_feat, ray_diff, mask, prompt, mean_hit_prob):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samplesblending_weights_valid, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]


        # rgb_feat: [n_rays, n_samples, n_views, n_feat] 
        rgb_feat1 = rgb_feat[:, :, :, None, :].repeat(1,1,1,num_views,1)  # [n_rays, n_samples, n_views, n_views, n_feat]
        rgb_feat2 = rgb_feat[:, :, None, :, :].repeat(1,1,num_views,1,1)  # [n_rays, n_samples, n_views, n_views, n_feat]

            # # rgb_feat_mat = rgb_feat1 *rgb_feat2
            # rgb_feat_sum = torch.sum(rgb_feat1 * rgb_feat2, dim =-1)
            # # rgb_feat_sum = F.cosine_similarity(rgb_feat1, rgb_feat2, dim=-1)
            # mask_rgb_feat_sum = torch.eye(num_views)[None, None, ...].type(torch.bool).to(rgb_feat_sum.device).repeat(rgb_feat_sum.shape[0], rgb_feat_sum.shape[1], 1, 1)
            # rgb_feat_sum[mask_rgb_feat_sum] = -1e8
            # rgb_feat_exp = torch.exp(rgb_feat_sum) # [n_rays, n_samples, n_views, n_views]
            

            # # rgb_feat_exp_mean_row = torch.mean(rgb_feat_exp, dim=3) # [n_rays, n_samples, n_views]
            # # rgb_feat_exp_mean_total = torch.mean(rgb_feat_exp_mean_row, dim=2)
            # rgb_feat_exp_sum_row = torch.sum(rgb_feat_exp, dim=3)
            # rgb_feat_exp_sum_total = torch.sum(rgb_feat_exp_sum_row, dim=2)
            # consistent_blending_weights = rgb_feat_exp_sum_row / rgb_feat_exp_sum_total.unsqueeze(-1) # [n_rays, n_samples, n_views]
            # rgb_feat_exp_sum_total = F.tanh(rgb_feat_exp_sum_total) # 0->1

            # assert torch.all(rgb_feat_exp_sum_total >=0) and torch.all(rgb_feat_exp_sum_total <= 1)

        # rgb_feat_mat = rgb_feat1 *rgb_feat2
        # rgb_feat_sum = torch.sum(rgb_feat1 * rgb_feat2, dim =-1)
        consistent_mats = F.cosine_similarity(rgb_feat1, rgb_feat2, dim=-1)
        mask_consistent_mats = torch.eye(num_views)[None, None, ...].type(torch.bool).to(consistent_mats.device).repeat(consistent_mats.shape[0], consistent_mats.shape[1], 1, 1)
        consistent_mats[mask_consistent_mats] = -1e8
        #rgb_feat_exp = torch.exp(rgb_feat_sum) # [n_rays, n_samples, n_views, n_views]

        # rgb_feat_exp_mean_row = torch.mean(rgb_feat_exp, dim=3) # [n_rays, n_samples, n_views]
        # rgb_feat_exp_mean_total = torch.mean(rgb_feat_exp_mean_row, dim=2)
        #rgb_feat_exp_sum_row = torch.sum(rgb_feat_exp*masks, dim=3)
        #rgb_feat_exp_sum_total = torch.sum(rgb_feat_exp_sum_row, dim=2)
        consistent_mats_exp = torch.exp(consistent_mats)
        consistent_mats_row = torch.log(torch.sum(consistent_mats_exp, dim=3)) # [n_rays, n_samples, n_views]
        consistent_mats_total = torch.log(torch.sum(consistent_mats_exp, dim=[2, 3]))
        # consistent_mats_row = torch.max(consistent_mats, dim=3)[0] # [n_rays, n_samples, n_views]
        # consistent_mats_total = torch.max(consistent_mats_row, dim=2)[0]
        # consistent_blending_weights = consistent_mats_row / consistent_mats_total.unsqueeze(-1)
        consistent_blending_weights = F.softmax(consistent_mats_row, dim=2)
        # consistent_weights = (consistent_mats_total +1) / 2.
        # consistent_weights = consistent_mats_total
        # # rgb_feat_max = torch.max(rgb_feat_sum,dim=[2,3])[0]
        # # rgb_feat_max = F.sigmoid(rgb_feat_max) # 0->1
        # print('debug', torch.max(consistent_weights), torch.min(consistent_weights), torch.mean(consistent_weights))
        # assert torch.all(consistent_weights >=0) and torch.all(consistent_weights <= 1)

        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat

            # if self.anti_alias_pooling:
            #     _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            #     exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            #     weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            #     weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8) # means it will trust the one more with more consistent view point
            # else:
            #     weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
        weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
        # prompt feats
        # prompt_rgb_feats = prompt[..., :35].squeeze(0) # [n_rays, n_samples, 3]
        # prompt_sigma_feats = prompt[..., 35:].squeeze(0) # [n_rays, n_samples, 16]
        # prompt_sigma_feats = prompt.squeeze(0)

        # neuray layer 0
        # weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight # [rn,dn,rfn,f]
        weight0 = consistent_blending_weights.unsqueeze(-1) * weight
        mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]
        mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        # x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1) 
        x = self.base_fc(x) # [n_rays, n_samples, n_views, 32]

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]

        globalfeat_clone = globalfeat.clone().detach()

        # NOTE combine weights
        # globalfeat = rgb_feat_exp_sum_total.unsqueeze(-1) * globalfeat + (1. - rgb_feat_exp_sum_total.unsqueeze(-1)) * prompt_sigma_feats
        # globalfeat = mean_hit_prob * globalfeat + (1. - mean_hit_prob) * prompt_sigma_feats
        # globalfeat = prompt_sigma_feats

        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding

        # print('ibr', globalfeat.shape, self.pos_encoding.shape)
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out1 = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # NOTE mask low consistent point by 0
        # sigma_out1 = sigma_out1.masked_fill(consistent_weights[..., None] < 0.2, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        # x = torch.cat([x, vis, ray_diff], dim=-1)
        # x = self.rgb_fc(x)
        # x = x.masked_fill(mask == 0, -1e9)
        # blending_weights_valid = F.softmax(x, dim=2)  # color blending

        # rgb_out1 = torch.sum(rgb_in*blending_weights_valid, dim=2)
        rgb_out1 = torch.sum(rgb_in * consistent_blending_weights.unsqueeze(-1), dim=2)
        # rgb_feat_blend = torch.sum(rgb_feat * blending_weights_valid, dim=2)
        # NOTE combine
        # rgb_feat_blend = mean_hit_prob * rgb_feat_blend + (1. - mean_hit_prob) * prompt_rgb_feats


        # rgb_out1 = self.rgb_final_fc(rgb_feat_blend)
        # rgb_feat_blend_clone = rgb_feat_blend.clone().detach()
        # gt_ibr = torch.cat([rgb_feat_blend_clone, globalfeat_clone], axis=-1)
        gt_ibr = globalfeat_clone

        # out = torch.cat([rgb_out, sigma_out1], dim=-1)

        # return out
        out1 = torch.cat([rgb_out1, sigma_out1], dim=-1)
        return out1, gt_ibr