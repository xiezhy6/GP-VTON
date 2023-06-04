import time
from options.train_options import TrainOptions
from models.networks import ResUnetGenerator, load_checkpoint_parallel
import torch.nn as nn
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
from tqdm import tqdm

opt = TrainOptions().parse()

def CreateDataset(opt):
    from data.aligned_dataset_vitonhd import AlignedDataset
    dataset = AlignedDataset()
    dataset.initialize(opt)
    return dataset

os.makedirs('sample',exist_ok=True)

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
    print('dataset size:', dataset_size)

gen_model = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.train()
gen_model.cuda()
load_checkpoint_parallel(gen_model, opt.PBAFN_gen_checkpoint)

gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen_model).to(device)
if opt.isTrain and len(opt.gpu_ids):
    model_gen = torch.nn.parallel.DistributedDataParallel(gen_model, device_ids=[opt.local_rank])


total_steps = (start_epoch-1) * dataset_size + epoch_iter

step = 0
step_per_batch = dataset_size


for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    # train_sampler.set_epoch(epoch)

    for data in tqdm(train_loader):

        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True

        person_clothes_edge = data['person_clothes_mask'].cuda()
        real_image = data['image'].cuda()
        clothes = data['color'].cuda()
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
        m_composite = m_composite * warped_prod_edge
        # m_composite =  person_clothes_edge.cuda()*m_composite1
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)


        ############## Display results and errors ##########
        path = 'sample/'+opt.name
        os.makedirs(path,exist_ok=True)

        a = real_image
        c = clothes
        k = p_tryon

        bz = pose.size(0)
        for bb in range(bz):
            # combine = torch.cat( [a[bb], c[bb],k[bb]], 2).squeeze()
            combine = k[bb].squeeze()
            # combine = torch.cat([a[bb], k[bb]], 2).squeeze()
        
            cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
            rgb = (cv_img*255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cloth_id = data['color_path'][bb].split('/')[-1]
            person_id = data['img_path'][bb].split('/')[-1]
            c_type = data['c_type'][bb]
            save_path = 'sample/'+opt.name+'/'+c_type+'___'+person_id+'___'+cloth_id
            cv2.imwrite(save_path, bgr)
    
    break