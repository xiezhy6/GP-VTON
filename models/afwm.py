import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from options.train_options import TrainOptions


from .correlation import correlation  # the custom cost volume layer
opt = TrainOptions().parse()


def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid(
        [torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
                 for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
                 for grid, size in zip(grid_list, reversed(sizes))]

    return torch.stack(grid_list, dim=-1)


def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))


def TVLoss_v2(x, mask):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    h, w = mask.size(2), mask.size(3)

    tv_h = tv_h * mask[:, :, :h-1, :]
    tv_w = tv_w * mask[:, :, :, :w-1]

    if torch.sum(mask) > 0:
        return (torch.sum(torch.abs(tv_h)) + torch.sum(torch.abs(tv_w))) / torch.sum(mask)
    else:
        return torch.sum(torch.abs(tv_h)) + torch.sum(torch.abs(tv_w))


def SquareTVLoss(flow):
    flow_x, flow_y = torch.split(flow, 1, dim=1)

    flow_x_diff_left = flow_x[:, :, :, 1:] - flow_x[:, :, :, :-1]
    flow_x_diff_right = flow_x[:, :, :, :-1] - flow_x[:, :, :, 1:]
    flow_x_diff_left = flow_x_diff_left[...,1:-1,:-1]
    flow_x_diff_right = flow_x_diff_right[...,1:-1,1:]

    flow_y_diff_top = flow_y[:, :, 1:, :] - flow_y[:, :, :-1, :]
    flow_y_diff_bottom = flow_y[:, :, :-1, :] - flow_y[:, :, 1:, :]
    flow_y_diff_top = flow_y_diff_top[...,:-1,1:-1]
    flow_y_diff_bottom = flow_y_diff_bottom[...,1:,1:-1]

    left_top_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_top))
    left_bottom_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_bottom))
    right_top_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_top))
    right_bottom_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_bottom))

    return torch.mean(left_top_diff+left_bottom_diff+right_top_diff+right_bottom_diff)

def SquareTVLoss_v2(flow, interval_list=[1,5]):
    flow_x, flow_y = torch.split(flow, 1, dim=1)

    tvloss = 0
    for interval in interval_list:
        flow_x_diff_left = flow_x[:, :, :, interval:] - flow_x[:, :, :, :-interval]
        flow_x_diff_right = flow_x[:, :, :, :-interval] - flow_x[:, :, :, interval:]
        flow_x_diff_left = flow_x_diff_left[...,interval:-interval,:-interval]
        flow_x_diff_right = flow_x_diff_right[...,interval:-interval,interval:]

        flow_y_diff_top = flow_y[:, :, interval:, :] - flow_y[:, :, :-interval, :]
        flow_y_diff_bottom = flow_y[:, :, :-interval, :] - flow_y[:, :, interval:, :]
        flow_y_diff_top = flow_y_diff_top[...,:-interval,interval:-interval]
        flow_y_diff_bottom = flow_y_diff_bottom[...,interval:,interval:-interval]

        left_top_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_top))
        left_bottom_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_bottom))
        right_top_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_top))
        right_bottom_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_bottom))

        tvloss += torch.mean(left_top_diff+left_bottom_diff+right_top_diff+right_bottom_diff)

    return tvloss


# backbone

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x) + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i-1], out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features


class RefinePyramid(nn.Module):
    def __init__(self, chns=[64, 128, 256, 256, 256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive
        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(
                fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x

        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + \
                    F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))


class AFlowNet_Vitonhd_lrarms(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super(AFlowNet_Vitonhd_lrarms, self).__init__()
        self.netLeftMain = []
        self.netTorsoMain = []
        self.netRightMain = []

        self.netLeftRefine = []
        self.netTorsoRefine = []
        self.netRightRefine = []

        self.netAttentionRefine = []
        self.netPartFusion = []
        self.netSeg = []

        for i in range(num_pyramid):
            netLeftMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netTorsoMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRightMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )

            netRefine_left_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRefine_torso_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRefine_right_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )

            netAttentionRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=4 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=3,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

            netSeg_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=fpn_dim*2, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=7,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

            partFusion_layer = torch.nn.Sequential(
                nn.Conv2d(fpn_dim*3, fpn_dim, kernel_size=1),
                ResBlock(fpn_dim)
            )

            self.netLeftMain.append(netLeftMain_layer)
            self.netTorsoMain.append(netTorsoMain_layer)
            self.netRightMain.append(netRightMain_layer)

            self.netLeftRefine.append(netRefine_left_layer)
            self.netTorsoRefine.append(netRefine_torso_layer)
            self.netRightRefine.append(netRefine_right_layer)

            self.netAttentionRefine.append(netAttentionRefine_layer)
            self.netPartFusion.append(partFusion_layer)
            self.netSeg.append(netSeg_layer)

        self.netLeftMain = nn.ModuleList(self.netLeftMain)
        self.netTorsoMain = nn.ModuleList(self.netTorsoMain)
        self.netRightMain = nn.ModuleList(self.netRightMain)

        self.netLeftRefine = nn.ModuleList(self.netLeftRefine)
        self.netTorsoRefine = nn.ModuleList(self.netTorsoRefine)
        self.netRightRefine = nn.ModuleList(self.netRightRefine)

        self.netAttentionRefine = nn.ModuleList(self.netAttentionRefine)
        self.netPartFusion = nn.ModuleList(self.netPartFusion)
        self.netSeg = nn.ModuleList(self.netSeg)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x, x_edge, x_full, x_edge_full, x_warps, x_conds, preserve_mask, warp_feature=True):
        last_flow = None
        last_flow_all = []
        delta_list = []
        x_all = []
        x_edge_all = []
        x_full_all = []
        x_edge_full_all = []
        attention_all = []
        seg_list = []
        delta_x_all = []
        delta_y_all = []
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array = torch.cuda.FloatTensor(weight_array).permute(3, 2, 0, 1)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        for i in range(len(x_warps)):
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_cond_concate = torch.cat([x_cond,x_cond,x_cond],0)
            x_warp_concate = torch.cat([x_warp,x_warp,x_warp],0)

            if last_flow is not None and warp_feature:
                x_warp_after = F.grid_sample(x_warp_concate, last_flow.detach().permute(0, 2, 3, 1),
                                             mode='bilinear', padding_mode='border')
            else:
                x_warp_after = x_warp_concate

            tenCorrelation = F.leaky_relu(input=correlation.FunctionCorrelation(
                tenFirst=x_warp_after, tenSecond=x_cond_concate, intStride=1), negative_slope=0.1, inplace=False)
            
            bz = x_cond.size(0)

            left_tenCorrelation = tenCorrelation[0:bz]
            torso_tenCorrelation = tenCorrelation[bz:2*bz]
            right_tenCorrelation = tenCorrelation[2*bz:]

            left_flow = self.netLeftMain[i](left_tenCorrelation)
            torso_flow = self.netTorsoMain[i](torso_tenCorrelation)
            right_flow = self.netRightMain[i](right_tenCorrelation)

            flow = torch.cat([left_flow,torso_flow,right_flow],0)

            delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')
            else:
                flow = flow.permute(0, 3, 1, 2)

            last_flow = flow
            x_warp_concate = F.grid_sample(x_warp_concate, flow.permute(
                0, 2, 3, 1), mode='bilinear', padding_mode='border')

            left_concat = torch.cat([x_warp_concate[0:bz], x_cond_concate[0:bz]], 1)
            torso_concat = torch.cat([x_warp_concate[bz:2*bz], x_cond_concate[bz:2*bz]],1)
            right_concat = torch.cat([x_warp_concate[2*bz:], x_cond_concate[2*bz:]],1)

            x_attention = torch.cat([x_warp_concate[0:bz],x_warp_concate[bz:2*bz],x_warp_concate[2*bz:],x_cond],1)
            fused_attention = self.netAttentionRefine[i](x_attention)
            fused_attention = self.softmax(fused_attention)

            left_flow = self.netLeftRefine[i](left_concat)
            torso_flow = self.netTorsoRefine[i](torso_concat)
            right_flow = self.netRightRefine[i](right_concat)   

            flow = torch.cat([left_flow,torso_flow,right_flow],0)
            delta_list.append(flow)
            flow = apply_offset(flow)
            flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')

            fused_flow = flow[0:bz] * fused_attention[:,0:1,...] + \
                         flow[bz:2*bz] * fused_attention[:,1:2,...] + \
                         flow[2*bz:] * fused_attention[:,2:3,...]
            last_fused_flow = F.interpolate(fused_flow, scale_factor=2, mode='bilinear')

            fused_attention = F.interpolate(fused_attention, scale_factor=2, mode='bilinear')
            attention_all.append(fused_attention)

            cur_x_full = F.interpolate(x_full, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_full_warp = F.grid_sample(cur_x_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_full_all.append(cur_x_full_warp)
            cur_x_edge_full = F.interpolate(x_edge_full, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_edge_full_warp = F.grid_sample(cur_x_edge_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_full_all.append(cur_x_edge_full_warp)

            last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)

            cur_x = F.interpolate(x, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_warp = F.grid_sample(cur_x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_all.append(cur_x_warp)
            cur_x_edge = F.interpolate(x_edge, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_warp_edge = F.grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_all.append(cur_x_warp_edge)

            flow_x, flow_y = torch.split(last_flow, 1, dim=1)
            delta_x = F.conv2d(flow_x, self.weight)
            delta_y = F.conv2d(flow_y, self.weight)
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

            # predict seg
            cur_preserve_mask = F.interpolate(preserve_mask, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_warp = torch.cat([x_warp,x_warp,x_warp],0)
            x_warp = F.interpolate(x_warp, scale_factor=2, mode='bilinear')
            x_cond = F.interpolate(x_cond, scale_factor=2, mode='bilinear')

            x_warp = F.grid_sample(x_warp, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')
            x_warp_left = x_warp[0:bz]
            x_warp_torso = x_warp[bz:2*bz]
            x_warp_right = x_warp[2*bz:]

            x_edge_left = cur_x_warp_edge[0:bz]
            x_edge_torso = cur_x_warp_edge[bz:2*bz]
            x_edge_right = cur_x_warp_edge[2*bz:]

            x_warp_left = x_warp_left * x_edge_left * (1-cur_preserve_mask)
            x_warp_torso = x_warp_torso * x_edge_torso * (1-cur_preserve_mask)
            x_warp_right = x_warp_right * x_edge_right * (1-cur_preserve_mask)

            x_warp = torch.cat([x_warp_left,x_warp_torso,x_warp_right],1)
            x_warp = self.netPartFusion[i](x_warp)

            concate = torch.cat([x_warp,x_cond],1)
            seg = self.netSeg[i](concate)
            seg_list.append(seg)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, x_full_all, \
                x_edge_full_all, attention_all, seg_list


class AFWM_Vitonhd_lrarms(nn.Module):
    def __init__(self, opt, input_nc, clothes_input_nc=3):
        super(AFWM_Vitonhd_lrarms, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        # num_filters = [64,128,256,512,512]
        fpn_dim = 256
        self.image_features = FeatureEncoder(clothes_input_nc+1, num_filters)
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        self.cond_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        
        self.aflow_net = AFlowNet_Vitonhd_lrarms(len(num_filters))
        self.old_lr = opt.lr
        self.old_lr_warp = opt.lr*0.2

    def forward(self, cond_input, image_input, image_edge, image_label_input, image_input_left, image_input_torso, \
                image_input_right, image_edge_left, image_edge_torso, image_edge_right, preserve_mask):
        image_input_concat = torch.cat([image_input, image_label_input],1)

        image_pyramids = self.image_FPN(self.image_features(image_input_concat))
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))  # maybe use nn.Sequential

        image_concat = torch.cat([image_input_left,image_input_torso,image_input_right],0)
        image_edge_concat = torch.cat([image_edge_left, image_edge_torso, image_edge_right],0)

        last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
            x_full_all, x_edge_full_all, attention_all, seg_list = self.aflow_net(image_concat, \
            image_edge_concat, image_input, image_edge, image_pyramids, cond_pyramids, \
            preserve_mask)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
                x_full_all, x_edge_full_all, attention_all, seg_list

    def update_learning_rate(self, optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate_warp(self, optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr

