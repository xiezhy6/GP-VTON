from options.train_options import TrainOptions
from models.networks import load_checkpoint_parallel
from models.afwm import AFWM_Vitonhd_lrarms, AFWM_Dressescode_lrarms
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
import tqdm

opt = TrainOptions().parse()
os.makedirs('sample/'+opt.name, exist_ok=True)

def CreateDataset(opt):
    if opt.dataset == 'vitonhd':
        from data.aligned_dataset_vitonhd import AlignedDataset
        dataset = AlignedDataset()
        dataset.initialize(opt, mode='test')
    elif opt.dataset == 'dresscode':
        from data.aligned_dataset_dresscode import AlignedDataset
        dataset = AlignedDataset()
        dataset.initialize(opt, mode='test', stage='warp')
    return dataset

torch.cuda.set_device(opt.local_rank)
torch.distributed.init_process_group(
    'nccl',
    init_method='env://'
)
device = torch.device(f'cuda:{opt.local_rank}')

train_data = CreateDataset(opt)
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=4, pin_memory=True, sampler=train_sampler)

if opt.dataset == 'vitonhd':
    warp_model = AFWM_Vitonhd_lrarms(opt, 51)
elif opt.dataset == 'dresscode':
    warp_model = AFWM_Dressescode_lrarms(opt, 51)
warp_model.train()
warp_model.cuda()
load_checkpoint_parallel(warp_model, opt.PBAFN_warp_checkpoint)

warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

if opt.isTrain and len(opt.gpu_ids):
    model = torch.nn.parallel.DistributedDataParallel(
        warp_model, device_ids=[opt.local_rank])

softmax = torch.nn.Softmax(dim=1)

for ii, data in enumerate(tqdm.tqdm(train_loader)):
    with torch.no_grad():
        pre_clothes_edge = data['edge']
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        pose = data['pose']

        size = data['color'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1,data['densepose'].data.long().cuda(),1.0)
        densepose = densepose * 2.0 - 1.0
        densepose_fore = data['densepose']/24.0

        left_cloth_sleeve_mask = data['flat_clothes_left_mask']
        cloth_torso_mask = data['flat_clothes_middle_mask']
        right_cloth_sleeve_mask = data['flat_clothes_right_mask']

        clothes_left = clothes * left_cloth_sleeve_mask
        clothes_torso = clothes * cloth_torso_mask
        clothes_right = clothes * right_cloth_sleeve_mask

        cloth_parse_for_d = data['flat_clothes_label'].cuda()
        pose = pose.cuda()
        clothes = clothes.cuda()
        clothes_left = clothes_left.cuda()
        clothes_torso = clothes_torso.cuda()
        clothes_right = clothes_right.cuda()
        pre_clothes_edge = pre_clothes_edge.cuda()
        left_cloth_sleeve_mask = left_cloth_sleeve_mask.cuda()
        cloth_torso_mask = cloth_torso_mask.cuda()
        right_cloth_sleeve_mask = right_cloth_sleeve_mask.cuda()
        preserve_mask3 = data['preserve_mask3'].cuda()

        if opt.resolution == 512:
            concat = torch.cat([densepose, pose, preserve_mask3], 1)
            if opt.dataset == 'vitonhd':
                flow_out = model(concat, clothes, pre_clothes_edge, cloth_parse_for_d, \
                                clothes_left, clothes_torso, clothes_right, \
                                left_cloth_sleeve_mask, cloth_torso_mask, right_cloth_sleeve_mask, \
                                preserve_mask3)
            elif opt.dataset == 'dresscode':
                cloth_type = data['flat_clothes_type'].cuda()
                flow_out = model(concat, clothes, pre_clothes_edge, cloth_parse_for_d, \
                                clothes_left, clothes_torso, clothes_right, \
                                left_cloth_sleeve_mask, cloth_torso_mask, right_cloth_sleeve_mask, \
                                preserve_mask3, cloth_type)

            last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
                x_full_all, x_edge_full_all, attention_all, seg_list = flow_out
        else:
            densepose_ds = F.interpolate(densepose, scale_factor=0.5, mode='nearest')
            pose_ds = F.interpolate(pose, scale_factor=0.5, mode='nearest')
            preserve_mask3_ds = F.interpolate(preserve_mask3, scale_factor=0.5, mode='nearest')
            concat = torch.cat([densepose_ds, pose_ds, preserve_mask3_ds], 1)

            clothes_ds = F.interpolate(clothes, scale_factor=0.5, mode='bilinear')
            pre_clothes_edge_ds = F.interpolate(pre_clothes_edge, scale_factor=0.5, mode='nearest')
            cloth_parse_for_d_ds = F.interpolate(cloth_parse_for_d, scale_factor=0.5, mode='nearest')
            clothes_left_ds = F.interpolate(clothes_left, scale_factor=0.5, mode='bilinear')
            clothes_torso_ds = F.interpolate(clothes_torso, scale_factor=0.5, mode='bilinear')
            clothes_right_ds = F.interpolate(clothes_right, scale_factor=0.5, mode='bilinear')
            left_cloth_sleeve_mask_ds = F.interpolate(left_cloth_sleeve_mask, scale_factor=0.5, mode='nearest')
            cloth_torso_mask_ds = F.interpolate(cloth_torso_mask, scale_factor=0.5, mode='nearest')
            right_cloth_sleeve_mask_ds = F.interpolate(right_cloth_sleeve_mask, scale_factor=0.5, mode='nearest')

            if opt.dataset == 'vitonhd':
                flow_out = model(concat, clothes_ds, pre_clothes_edge_ds, cloth_parse_for_d_ds, \
                                clothes_left_ds, clothes_torso_ds, clothes_right_ds, \
                                left_cloth_sleeve_mask_ds, cloth_torso_mask_ds, right_cloth_sleeve_mask_ds, \
                                preserve_mask3_ds)
            elif opt.dataset == 'dresscode':
                cloth_type = data['flat_clothes_type'].cuda()
                cloth_type_ds = F.interpolate(cloth_type, scale_factor=0.5, mode='bilinear')
                flow_out = model(concat, clothes_ds, pre_clothes_edge_ds, cloth_parse_for_d_ds, \
                                clothes_left_ds, clothes_torso_ds, clothes_right_ds, \
                                left_cloth_sleeve_mask_ds, cloth_torso_mask_ds, right_cloth_sleeve_mask_ds, \
                                preserve_mask3_ds, cloth_type_ds)
                
            last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
                x_full_all, x_edge_full_all, attention_all, seg_list = flow_out
            last_flow =  F.interpolate(last_flow, scale_factor=2, mode='bilinear')

        bz = pose.size(0)

        left_last_flow = last_flow[0:bz]
        torso_last_flow = last_flow[bz:2*bz]
        right_last_flow = last_flow[2*bz:]

        left_warped_full_cloth = F.grid_sample(clothes_left.cuda(), left_last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros')
        torso_warped_full_cloth = F.grid_sample(clothes_torso.cuda(), torso_last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros')
        right_warped_full_cloth = F.grid_sample(clothes_right.cuda(), right_last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros')

        left_warped_cloth_edge = F.grid_sample(left_cloth_sleeve_mask.cuda(), left_last_flow.permute(0, 2, 3, 1),mode='nearest', padding_mode='zeros')
        torso_warped_cloth_edge = F.grid_sample(cloth_torso_mask.cuda(), torso_last_flow.permute(0, 2, 3, 1),mode='nearest', padding_mode='zeros')
        right_warped_cloth_edge = F.grid_sample(right_cloth_sleeve_mask.cuda(), right_last_flow.permute(0, 2, 3, 1),mode='nearest', padding_mode='zeros')

        for bb in range(bz):
            seg_preds = torch.argmax(softmax(seg_list[-1]),dim=1)[:,None,...].float()
            if opt.resolution == 1024:
                seg_preds = F.interpolate(seg_preds, scale_factor=2, mode='nearest')

            c_type = data['c_type'][bb]

            if opt.dataset == 'vitonhd':
                left_mask = (seg_preds[bb]==1).float()
                torso_mask = (seg_preds[bb]==2).float()
                right_mask = (seg_preds[bb]==3).float()

                left_arm_mask = (seg_preds[bb]==4).float()
                right_arm_mask = (seg_preds[bb]==5).float()
                neck_mask = (seg_preds[bb]==6).float()

                warped_cloth_fusion = left_warped_full_cloth[bb] * left_mask + \
                                    torso_warped_full_cloth[bb] * torso_mask + \
                                    right_warped_full_cloth[bb] * right_mask
                
                warped_edge_fusion = left_warped_cloth_edge[bb] * left_mask * 1 + \
                                        torso_warped_cloth_edge[bb] * torso_mask * 2 + \
                                        right_warped_cloth_edge[bb] * right_mask * 3

                warped_cloth_fusion = warped_cloth_fusion * (1-preserve_mask3[bb])
                warped_edge_fusion = warped_edge_fusion * (1-preserve_mask3[bb])
                
                warped_edge_fusion = warped_edge_fusion + \
                                        left_arm_mask * 4 + \
                                        right_arm_mask * 5 + \
                                        neck_mask * 6
            elif opt.dataset == 'dresscode':
                if c_type == 'upper' or c_type == 'dresses':
                    left_mask = (seg_preds[bb]==1).float()
                    torso_mask = (seg_preds[bb]==2).float()
                    right_mask = (seg_preds[bb]==3).float()
                else:
                    left_mask = (seg_preds[bb]==4).float()
                    torso_mask = (seg_preds[bb]==5).float()
                    right_mask = (seg_preds[bb]==6).float()

                left_arms_mask = (seg_preds[bb]==7).float()
                right_arms_mask = (seg_preds[bb]==8).float()
                neck_mask = (seg_preds[bb]==9).float()

                warped_cloth_fusion = left_warped_full_cloth[bb] * left_mask + \
                                    torso_warped_full_cloth[bb] * torso_mask + \
                                    right_warped_full_cloth[bb] * right_mask

                if c_type == 'upper' or c_type == 'dresses':
                    warped_edge_fusion = left_warped_cloth_edge[bb] * left_mask * 1 + \
                                        torso_warped_cloth_edge[bb] * torso_mask * 2 + \
                                        right_warped_cloth_edge[bb] * right_mask * 3
                else:
                    warped_edge_fusion = left_warped_cloth_edge[bb] * left_mask * 4 + \
                                        torso_warped_cloth_edge[bb] * torso_mask * 5 + \
                                        right_warped_cloth_edge[bb] * right_mask * 6

                warped_cloth_fusion = warped_cloth_fusion * (1-preserve_mask3[bb])
                warped_edge_fusion = warped_edge_fusion * (1-preserve_mask3[bb])

                warped_edge_fusion = warped_edge_fusion + \
                                        left_arms_mask * 7 + \
                                        right_arms_mask * 8 + \
                                        neck_mask * 9

            eee = warped_cloth_fusion
            eee_edge = torch.cat([warped_edge_fusion,warped_edge_fusion,warped_edge_fusion],0)
            eee_edge = eee_edge.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

            cv_img = (eee.permute(1, 2, 0).detach().cpu().numpy()+1)/2
            rgb = (cv_img*255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            bgr = np.concatenate([bgr,eee_edge],1)

            cloth_id = data['color_path'][bb].split('/')[-1]
            person_id = data['img_path'][bb].split('/')[-1]
            save_path = 'sample/'+opt.name+'/'+c_type+'___'+person_id+'___'+cloth_id[:-4]+'.png'
            cv2.imwrite(save_path, bgr)
