import time
from options.train_options import TrainOptions
from models.networks import VGGLoss, save_checkpoint, load_checkpoint_parallel, SpectralDiscriminator, GANLoss, set_requires_grad
from models.afwm import TVLoss, AFWM_Dressescode_lrarms
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import cv2
import datetime
import functools


opt = TrainOptions().parse()
path = 'runs/'+opt.name
os.makedirs(path, exist_ok=True)

os.makedirs('sample', exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

def CreateDataset(opt):
    from data.aligned_dataset_dresscode import AlignedDataset
    dataset = AlignedDataset()
    dataset.initialize(opt, mode='train', stage='warp')
    return dataset

torch.cuda.set_device(opt.local_rank)
torch.distributed.init_process_group(
    'nccl',
    init_method='env://'
)
device = torch.device(f'cuda:{opt.local_rank}')

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=4, pin_memory=True, sampler=train_sampler)
dataset_size = len(train_loader)
if opt.local_rank == 0:
    print('#training images = %d' % dataset_size)

warp_model = AFWM_Dressescode_lrarms(opt, 51)
warp_model.train()
warp_model.cuda()
if opt.PBAFN_warp_checkpoint is not None:
    load_checkpoint_parallel(warp_model, opt.PBAFN_warp_checkpoint)
warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)
if opt.isTrain and len(opt.gpu_ids):
    model = torch.nn.parallel.DistributedDataParallel(
        warp_model, device_ids=[opt.local_rank])

params_warp = [p for p in model.parameters()]
optimizer_warp = torch.optim.Adam(
    params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))


discriminator = SpectralDiscriminator(opt, input_nc=62, ndf=64, n_layers=3,
                                        norm_layer=functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True), use_sigmoid=False)
discriminator.train()
discriminator.cuda()
if opt.pretrain_checkpoint_D is not None:
    load_checkpoint_parallel(discriminator, opt.pretrain_checkpoint_D)

discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator).to(device)
if opt.isTrain and len(opt.gpu_ids):
    discriminator = torch.nn.parallel.DistributedDataParallel(
        discriminator, device_ids=[opt.local_rank])

params_D = list(filter(lambda p: p.requires_grad,
                discriminator.parameters()))
optimizer_D = torch.optim.Adam(
    params_D, lr=opt.lr_D, betas=(opt.beta1, 0.999))

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()
criterionLSGANloss = GANLoss().cuda()
softmax = torch.nn.Softmax(dim=1)

total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size

if opt.local_rank == 0:
    writer = SummaryWriter(path)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    train_sampler.set_epoch(epoch)

    if opt.local_rank == 0:
        path = 'sample/'+opt.name
        os.makedirs(path, exist_ok=True)

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True

        pre_clothes_edge = data['edge']
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        person_clothes_edge = data['person_clothes_mask']
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        
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

        part_mask = torch.cat([left_cloth_sleeve_mask,cloth_torso_mask,right_cloth_sleeve_mask],0)
        part_mask = (torch.sum(part_mask,dim=(2,3),keepdim=True)>0).float().cuda()

        clothes_left = clothes * left_cloth_sleeve_mask
        clothes_torso = clothes * cloth_torso_mask
        clothes_right = clothes * right_cloth_sleeve_mask
 
        cloth_parse_for_d = data['flat_clothes_label'].cuda()
        cloth_parse_vis = torch.cat([cloth_parse_for_d,cloth_parse_for_d,cloth_parse_for_d],1)

        person_clothes_left_sleeve_mask = data['person_clothes_left_mask']
        person_clothes_torso_mask = data['person_clothes_middle_mask']
        person_clothes_right_sleeve_mask = data['person_clothes_right_mask']

        person_clothes_mask_concate = torch.cat([person_clothes_left_sleeve_mask,person_clothes_torso_mask,person_clothes_right_sleeve_mask],0)

        seg_label_tensor = data['seg_gt']
        seg_gt_tensor = (seg_label_tensor / 9 * 2 -1).float()
        seg_label_onehot_tensor = data['seg_gt_onehot'] * 2 - 1.0

        seg_label_tensor = seg_label_tensor.cuda()
        seg_gt_tensor = seg_gt_tensor.cuda()
        seg_label_onehot_tensor = seg_label_onehot_tensor.cuda()

        person_clothes = person_clothes.cuda()
        person_clothes_edge = person_clothes_edge.cuda()
        pose = pose.cuda()

        clothes = clothes.cuda()
        clothes_left = clothes_left.cuda()
        clothes_torso = clothes_torso.cuda()
        clothes_right = clothes_right.cuda()
        pre_clothes_edge = pre_clothes_edge.cuda()
        left_cloth_sleeve_mask = left_cloth_sleeve_mask.cuda()
        cloth_torso_mask = cloth_torso_mask.cuda()
        right_cloth_sleeve_mask = right_cloth_sleeve_mask.cuda()

        person_clothes_left_sleeve_mask = person_clothes_left_sleeve_mask.cuda()
        person_clothes_torso_mask = person_clothes_torso_mask.cuda()
        person_clothes_right_sleeve_mask = person_clothes_right_sleeve_mask.cuda()
        person_clothes_mask_concate = person_clothes_mask_concate.cuda()
        person_clothes_left_sleeve = person_clothes * person_clothes_left_sleeve_mask
        person_clothes_torso = person_clothes * person_clothes_torso_mask
        person_clothes_right_sleeve = person_clothes * person_clothes_right_sleeve_mask

        preserve_mask = data['preserve_mask'].cuda()
        preserve_mask2 = data['preserve_mask2'].cuda()
        preserve_mask3 = data['preserve_mask3'].cuda()

        preserve_legs_mask = data['preserve_legs_mask'].cuda()
        preserve_left_pants_mask = data['preserve_left_pants_mask'].cuda()
        preserve_right_pants_mask = data['preserve_right_pants_mask'].cuda()

        cloth_type = data['flat_clothes_type'].cuda()

        concat = torch.cat([densepose, pose, preserve_mask3], 1)
        flow_out = model(concat, clothes, pre_clothes_edge, cloth_parse_for_d, \
                        clothes_left, clothes_torso, clothes_right, \
                        left_cloth_sleeve_mask, cloth_torso_mask, right_cloth_sleeve_mask, \
                        preserve_mask3, cloth_type)

        last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
            x_full_all, x_edge_full_all, attention_all, seg_list = flow_out

        # update parsing discriminator
        set_requires_grad(discriminator, True)
        optimizer_D.zero_grad()
        pred_seg_D = seg_list[-1]
        D_concat = torch.cat([concat, cloth_parse_for_d.cuda()],1)
        D_in_fake = torch.cat([D_concat, pred_seg_D.detach()], 1)
        D_in_real = torch.cat([D_concat, seg_label_onehot_tensor], 1)
        loss_gan_D = (criterionLSGANloss(discriminator(
            D_in_fake), False) + criterionLSGANloss(discriminator(D_in_real), True)) * 0.5 * 0.1
        loss_gan_D.backward()
        optimizer_D.step()
        set_requires_grad(discriminator, False)

        D_in_fake_G = torch.cat([D_concat, pred_seg_D], 1)
        loss_gan_G = criterionLSGANloss(
            discriminator(D_in_fake_G), True)* 0.1

        bz = pose.size(0)

        epsilon = 0.001
        loss_smooth = sum([TVLoss(x*part_mask) for x in delta_list])

        loss_all = 0
        loss_l1_total = 0
        loss_vgg_total = 0
        loss_edge_total = 0
        loss_second_smooth_total = 0

        loss_full_l1_total = 0
        loss_full_vgg_total = 0
        loss_full_edge_total = 0

        loss_attention_total = 0
        loss_seg_ce_total = 0

        softmax = torch.nn.Softmax(dim=1)
        class_weight  = torch.FloatTensor([1,100,15,100,50,100,50,70,70,150]).cuda()
        criterionCE = nn.CrossEntropyLoss(weight=class_weight)

        for num in range(5):
            cur_seg_label_tensor = F.interpolate(
                seg_label_tensor, scale_factor=0.5**(4-num), mode='nearest').cuda()

            pred_seg = seg_list[num]
            loss_seg_ce = criterionCE(pred_seg, cur_seg_label_tensor.long()[:,0,...])

            pred_attention = attention_all[num]
            pred_mask_concate = torch.cat([pred_attention[:,0:1,...],pred_attention[:,1:2,...],pred_attention[:,2:3,...]],0)
            cur_person_clothes_mask_gt = F.interpolate(person_clothes_mask_concate, scale_factor=0.5**(4-num), mode='bilinear')

            loss_attention = criterionL1(pred_mask_concate,cur_person_clothes_mask_gt)

            cur_person_clothes_left_sleeve = F.interpolate(person_clothes_left_sleeve, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_left_sleeve_mask = F.interpolate(person_clothes_left_sleeve_mask, scale_factor=0.5**(4-num), mode='bilinear')
            
            cur_person_clothes_torso = F.interpolate(person_clothes_torso, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_torso_mask = F.interpolate(person_clothes_torso_mask, scale_factor=0.5**(4-num), mode='bilinear')
            
            cur_person_clothes_right_sleeve = F.interpolate(person_clothes_right_sleeve, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_right_sleeve_mask = F.interpolate(person_clothes_right_sleeve_mask, scale_factor=0.5**(4-num), mode='bilinear')

            cur_person_clothes = torch.cat([cur_person_clothes_left_sleeve, cur_person_clothes_torso, cur_person_clothes_right_sleeve],0)
            cur_person_clothes_edge = torch.cat([cur_person_clothes_left_sleeve_mask, cur_person_clothes_torso_mask, cur_person_clothes_right_sleeve_mask],0)

            pred_clothes = x_all[num]
            pred_edge = x_edge_all[num]

            cur_preserve_mask = F.interpolate(preserve_mask, scale_factor=0.5**(4-num), mode='bilinear')
            cur_preserve_mask2 = F.interpolate(preserve_mask2, scale_factor=0.5**(4-num), mode='bilinear')

            cur_preserve_legs_mask = F.interpolate(preserve_legs_mask, scale_factor=0.5**(4-num), mode='bilinear')
            cur_preserve_left_pants_mask = F.interpolate(preserve_left_pants_mask, scale_factor=0.5**(4-num), mode='bilinear')
            cur_preserve_right_pants_mask = F.interpolate(preserve_right_pants_mask, scale_factor=0.5**(4-num), mode='bilinear')

            cur_preserve_mask_concate = torch.cat([cur_preserve_mask,cur_preserve_mask2,cur_preserve_mask],0)
            cur_person_clothes_mask_concate = torch.cat([cur_person_clothes_torso_mask,cur_person_clothes_left_sleeve_mask+cur_person_clothes_right_sleeve_mask,cur_person_clothes_torso_mask],0)
            cur_lower_preserve_mask_concate = torch.cat([cur_preserve_right_pants_mask,cur_preserve_legs_mask,cur_preserve_left_pants_mask],0)

            cur_preserve_mask_concate = cur_preserve_mask_concate + cur_person_clothes_mask_concate + cur_lower_preserve_mask_concate
            cur_preserve_mask_concate = (cur_preserve_mask_concate>0).float()

            if epoch > opt.mask_epoch:
                pred_clothes = pred_clothes * (1-cur_preserve_mask_concate)
                pred_edge = pred_edge * (1-cur_preserve_mask_concate)

            loss_l1 = criterionL1(pred_clothes*part_mask, cur_person_clothes*part_mask)
            loss_vgg = criterionVGG(pred_clothes*part_mask, cur_person_clothes*part_mask)
            loss_edge = criterionL1(pred_edge*part_mask, cur_person_clothes_edge*part_mask)

            cur_person_clothes_full = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_edge_full = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')

            pred_clothes_full = x_full_all[num]
            pred_edge_full = x_edge_full_all[num]

            if epoch > opt.mask_epoch:
                pred_clothes_full = pred_clothes_full * (1-cur_preserve_mask2)
                pred_edge_full = pred_edge_full * (1-cur_preserve_mask2)

            loss_full_l1 = criterionL1(pred_clothes_full, cur_person_clothes_full)
            loss_full_edge = criterionL1(pred_edge_full, cur_person_clothes_edge_full)

            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon*epsilon).pow(0.45)
            loss_flow_x = loss_flow_x * part_mask
            loss_flow_x = torch.sum(loss_flow_x[0:int(b/3)]) / (int(b/3)*c*h*w) + \
                          40 * torch.sum(loss_flow_x[int(b/3):int(b/3)*2]) / (int(b/3)*c*h*w) + \
                          torch.sum(loss_flow_x[int(b/3)*2:]) / (int(b/3)*c*h*w)
            loss_flow_x = loss_flow_x / 3

            loss_flow_y = (delta_y_all[num].pow(2) + epsilon*epsilon).pow(0.45)
            loss_flow_y = loss_flow_y * part_mask
            loss_flow_y = torch.sum(loss_flow_y[0:int(b/3)]) / (int(b/3)*c*h*w) + \
                          40 * torch.sum(loss_flow_y[int(b/3):int(b/3)*2]) / (int(b/3)*c*h*w) + \
                          torch.sum(loss_flow_y[int(b/3)*2:]) / (int(b/3)*c*h*w)
            loss_flow_y = loss_flow_y / 3
            loss_second_smooth = loss_flow_x + loss_flow_y

            loss_all = loss_all + (num+1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + \
                (num+1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth + \
                (num+1) * loss_full_l1 + \
                (num+1) * 2 * loss_full_edge + \
                (num+1) * loss_attention * 0.5 + \
                (num+1) * loss_seg_ce * 0.5

            loss_l1_total += loss_l1 * (num + 1)
            loss_vgg_total += loss_vgg * (num + 1) * 0.2
            loss_edge_total += loss_edge * (num + 1) * 2
            loss_second_smooth_total += loss_second_smooth * (num + 1) * 6

            loss_full_l1_total += (num+1) * loss_full_l1
            loss_full_edge_total += (num+1) * 2 * loss_full_edge
            loss_attention_total += (num+1) * loss_attention * 0.5
            loss_seg_ce_total += loss_seg_ce * (num+1) * 0.5

        loss_all = opt.first_order_smooth_weight * loss_smooth + loss_all + \
                    loss_gan_G

        if step % opt.write_loss_frep == 0:
            if opt.local_rank == 0:
                writer.add_scalar('loss_all', loss_all, step)
                writer.add_scalar('loss_l1', loss_l1_total, step)
                writer.add_scalar('loss_vgg', loss_vgg_total, step)
                writer.add_scalar('loss_edge', loss_edge_total, step)
                writer.add_scalar('loss_second_smooth',
                                  loss_second_smooth_total, step)
                writer.add_scalar('loss_smooth', loss_smooth *
                                  opt.first_order_smooth_weight, step)
                writer.add_scalar('loss_full_l1', loss_full_l1_total, step)
                writer.add_scalar('loss_full_edge', loss_full_edge_total, step)
                writer.add_scalar('loss_attention', loss_attention_total, step)
                writer.add_scalar('loss_seg_ce', loss_seg_ce_total, step)

        optimizer_warp.zero_grad()
        loss_all.backward()
        optimizer_warp.step()
        ############## Display results and errors ##########
        
        bz = real_image.size(0)
        
        warped_cloth = x_all[4]
        left_warped_cloth = warped_cloth[0:bz]
        torso_warped_cloth = warped_cloth[bz:2*bz]
        right_warped_cloth = warped_cloth[2*bz:]

        warped_cloth = left_warped_cloth + torso_warped_cloth + right_warped_cloth

        if step % opt.display_freq == 0:
            if opt.local_rank == 0:
                a = real_image.float().cuda()
                b = person_clothes.cuda()
                c = clothes.cuda()
                d = torch.cat([densepose_fore.cuda(),densepose_fore.cuda(),densepose_fore.cuda()],1)
                cm = cloth_parse_vis.cuda()
                warped_cloth = warped_cloth * (1-preserve_mask)
                e = warped_cloth

                bz = pose.size(0)

                seg_preds = torch.argmax(softmax(seg_list[-1]),dim=1)[:,None,...].float()
                c_type = data['c_type'][0]
                if c_type == 'upper' or c_type == 'dresses':
                    left_mask = (seg_preds==1).float()
                    torso_mask = (seg_preds==2).float()
                    right_mask = (seg_preds==3).float()
                else:
                    left_mask = (seg_preds==4).float()
                    torso_mask = (seg_preds==5).float()
                    right_mask = (seg_preds==6).float()
                warped_cloth_fusion = left_warped_cloth * left_mask + \
                                    torso_warped_cloth * torso_mask + \
                                    right_warped_cloth * right_mask
                warped_cloth_fusion = warped_cloth_fusion *  (1-preserve_mask)
                eee = warped_cloth_fusion

                vis_pose = (pose > 0).float()
                vis_pose = torch.sum(vis_pose.cuda(), dim=1).unsqueeze(1)
                vis_pose_mask = (vis_pose > 0).to(
                    vis_pose.device).to(vis_pose.dtype)
                g = torch.cat([vis_pose, vis_pose, vis_pose], 1)

                h = torch.cat([preserve_mask, preserve_mask, preserve_mask], 1)
                h2 = torch.cat([preserve_mask2, preserve_mask2, preserve_mask2], 1)

                seg_gt_vis = torch.cat([seg_gt_tensor,seg_gt_tensor,seg_gt_tensor],1).cuda()
                seg_preds = torch.argmax(softmax(seg_list[-1]),dim=1)[:,None,...].float()
                seg_preds = seg_preds / 9 * 2 - 1

                seg_preds_vis = torch.cat([seg_preds,seg_preds,seg_preds],1)
                
                preserve_left_pants_mask_vis = torch.cat([preserve_left_pants_mask,preserve_left_pants_mask,preserve_left_pants_mask],1)
                preserve_right_pants_mask_vis = torch.cat([preserve_right_pants_mask,preserve_right_pants_mask,preserve_right_pants_mask],1)
                preserve_legs_mask_vis = torch.cat([preserve_legs_mask,preserve_legs_mask,preserve_legs_mask],1)

                combine = torch.cat([a[0], c[0], cm[0], g[0], d[0], h[0], right_warped_cloth[0], torso_warped_cloth[0], \
                                    left_warped_cloth[0], e[0], eee[0], \
                                    b[0], seg_preds_vis[0], seg_gt_vis[0]], 2).squeeze()

                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
                writer.add_image('combine', (combine.data + 1) / 2.0, step)
                rgb = (cv_img*255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite('sample/'+opt.name+'/' +
                            str(epoch).zfill(3)+'_'+str(step)+'.jpg', bgr)

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch-step % step_per_batch) + \
            step_per_batch*(opt.niter + opt.niter_decay-epoch)
        eta = iter_delta_time*step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        if step % opt.print_freq == 0:
            if opt.local_rank == 0:
                print('{}:{}:[step-{}]--[loss-{:.6f}]--[learning rate-{}]--[ETA-{}]'.format(
                    now, epoch_iter, step, loss_all, model.module.old_lr, eta))

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    if opt.local_rank == 0:
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        if opt.local_rank == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            save_checkpoint(model.module, os.path.join(
                opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch+1)))
            save_checkpoint(discriminator.module, os.path.join(
                        opt.checkpoints_dir, opt.name, 'PBAFN_D_epoch_%03d.pth' % (epoch+1)))

    if epoch > opt.niter:
        model.module.update_learning_rate(optimizer_warp)
        discriminator.module.update_learning_rate(optimizer_D, opt)
