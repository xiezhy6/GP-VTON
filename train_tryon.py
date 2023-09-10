import time
from options.train_options import TrainOptions
from models.networks import ResUnetGenerator, VGGLoss, save_checkpoint, SpectralDiscriminator, GANLoss, set_requires_grad, load_checkpoint_parallel
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

def CreateDataset(opt):
    if opt.dataset == 'vitonhd':
        from data.aligned_dataset_vitonhd import AlignedDataset
        dataset = AlignedDataset()
        dataset.initialize(opt)
    elif opt.dataset == 'dresscode':
        from data.aligned_dataset_dresscode import AlignedDataset
        dataset = AlignedDataset()
        dataset.initialize(opt, mode='train', stage='gen')
    return dataset

opt = TrainOptions().parse()

run_path = 'runs/'+opt.name
sample_path = 'sample/'+opt.name
os.makedirs(run_path, exist_ok=True)
os.makedirs(sample_path, exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

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
dataset_size = len(train_loader)

gen_model = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.train()
gen_model.cuda()
if opt.PBAFN_gen_checkpoint is not None:
    load_checkpoint_parallel(gen_model, opt.PBAFN_gen_checkpoint)
gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen_model).to(device)
if opt.isTrain and len(opt.gpu_ids):
    model_gen = torch.nn.parallel.DistributedDataParallel(gen_model, device_ids=[opt.local_rank])

params_gen = [p for p in model_gen.parameters()]
optimizer_gen = torch.optim.Adam(params_gen, lr=opt.lr, betas=(opt.beta1, 0.999))

discriminator = SpectralDiscriminator(opt, input_nc=39, ndf=64, n_layers=3,
                                      norm_layer=functools.partial(nn.BatchNorm2d, 
                                      affine=True, track_running_stats=True), use_sigmoid=False)
discriminator.train()
discriminator.cuda()
if opt.pretrain_checkpoint_D is not None:
    load_checkpoint_parallel(discriminator, opt.pretrain_checkpoint_D)
discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator).to(device)
if opt.isTrain and len(opt.gpu_ids):
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[opt.local_rank])

params_D = list(filter(lambda p: p.requires_grad,
                discriminator.parameters()))
optimizer_D = torch.optim.Adam(
    params_D, lr=opt.lr_D, betas=(opt.beta1, 0.999))

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()
criterionLSGANloss = GANLoss().cuda()

if opt.local_rank == 0:
    writer = SummaryWriter(run_path)
    print('#training images = %d' % dataset_size)

start_epoch, epoch_iter = 1, 0
total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    train_sampler.set_epoch(epoch)

    for ii, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1

        person_clothes_edge = data['person_clothes_mask'].cuda()
        real_image = data['image'].cuda()
        preserve_mask = data['preserve_mask3'].cuda()
        preserve_region = real_image * preserve_mask
        warped_cloth = data['warped_cloth'].cuda()
        warped_prod_edge = data['warped_edge'].cuda()
        arms_color = data['arms_color'].cuda()
        arms_neck_label= data['arms_neck_lable'].cuda()
        pose = data['pose'].cuda()

        gen_inputs = torch.cat([preserve_region, warped_cloth, warped_prod_edge, arms_neck_label, arms_color, pose], 1)

        gen_outputs = model_gen(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite1 = m_composite * warped_prod_edge
        if opt.dataset == 'vitonhd':
            m_composite =  person_clothes_edge.cuda()*m_composite1
        elif opt.dataset == 'dresscode':
            m_composite =  m_composite1
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        set_requires_grad(discriminator, True)
        optimizer_D.zero_grad()
        pred_seg_D = p_rendered
        D_in_fake = torch.cat([gen_inputs, pred_seg_D.detach()], 1)
        D_in_real = torch.cat([gen_inputs, real_image], 1)
        loss_gan_D = (criterionLSGANloss(discriminator(
            D_in_fake), False) + criterionLSGANloss(discriminator(D_in_real), True)) * 0.5
        loss_gan_D.backward()
        optimizer_D.step()
        set_requires_grad(discriminator, False)

        D_in_fake_G = torch.cat([gen_inputs, pred_seg_D], 1)
        loss_gan_G = criterionLSGANloss(discriminator(D_in_fake_G), True)* 0.5

        loss_mask_l1 = torch.mean(torch.abs(1 - m_composite)) * 5
        loss_l1 = criterionL1(p_tryon, real_image.cuda())
        loss_vgg = criterionVGG(p_tryon,real_image.cuda())
        bg_loss_l1 = criterionL1(p_rendered, real_image.cuda())
        bg_loss_vgg = criterionVGG(p_rendered, real_image.cuda())
        gen_loss = (loss_l1 * 5 + loss_vgg + bg_loss_l1 * 5 + bg_loss_vgg + loss_mask_l1)

        if step % opt.write_loss_frep == 0:
            if opt.local_rank == 0:
                writer.add_scalar('gen_loss', gen_loss, step)
                writer.add_scalar('gen_mask_l1_loss', loss_mask_l1 * 1.0, step)
                writer.add_scalar('gen_l1_loss', loss_l1 * 5, step)
                writer.add_scalar('gen_vgg_loss', loss_vgg, step)
                writer.add_scalar('gen_bg_l1_loss', bg_loss_l1 * 5, step)
                writer.add_scalar('gen_bg_vgg_loss', bg_loss_vgg, step)
                writer.add_scalar('gen_GAN_G_loss', loss_gan_G, step)
                writer.add_scalar('gen_GAN_D_loss', loss_gan_D, step)

        loss_all =  gen_loss + loss_gan_G

        optimizer_gen.zero_grad()
        loss_all.backward()
        optimizer_gen.step()

        ############## Display results and errors ##########
        if step % opt.display_freq == 0:
            if opt.local_rank == 0:
                a = real_image.float().cuda()
                e = warped_cloth
                f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
                ff = arms_color
                g = preserve_region.cuda()
                vis_pose = (pose > 0).float()
                vis_pose = torch.sum(vis_pose.cuda(), dim=1).unsqueeze(1)
                vis_pose_mask = (vis_pose > 0).to(
                    vis_pose.device).to(vis_pose.dtype)
                h = torch.cat([vis_pose, vis_pose, vis_pose], 1)
                i = p_rendered
                j = torch.cat([m_composite1, m_composite1, m_composite1], 1)
                k = p_tryon

                l = torch.cat([arms_neck_label,arms_neck_label,arms_neck_label],1)

                combine = torch.cat(
                    [a[0], h[0], g[0], f[0], l[0], ff[0], e[0], j[0], i[0], k[0]], 2).squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
                writer.add_image('combine', (combine.data + 1) / 2.0, step)
                rgb = (cv_img*255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite('sample/'+opt.name+'/'+str(epoch) +
                            '_'+str(step)+'.jpg', bgr)

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch-step%step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
        eta = iter_delta_time*step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')

        if step % opt.print_freq == 0:
            if opt.local_rank == 0:
                print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(
                    now, epoch_iter, step, loss_all, eta))

        if epoch_iter >= dataset_size:
            break

    iter_end_time = time.time()
    if opt.local_rank == 0:
      print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
      if opt.local_rank == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        save_checkpoint(model_gen.module, os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_gen_epoch_%03d.pth' % (epoch+1)))
        save_checkpoint(discriminator.module, os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_D_epoch_%03d.pth' % (epoch+1)))
    if epoch > opt.niter:
        discriminator.module.update_learning_rate_warp(optimizer_D)
        model_gen.module.update_learning_rate(optimizer_gen)
