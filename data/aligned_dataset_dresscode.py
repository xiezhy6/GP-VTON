import os
from random import random
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import torch
import numpy as np
from PIL import ImageDraw
import cv2
import pycocotools.mask as maskUtils
import math

class AlignedDataset(BaseDataset):
    def initialize(self, opt, mode='train', stage='warp'):
        self.opt = opt
        self.root = opt.dataroot
        self.warproot = opt.warproot
        self.resolution = opt.resolution
        self.stage = stage

        if self.resolution == 512:
            self.fine_height=512
            self.fine_width=384
            self.radius=8
        else:
            self.fine_height=1024
            self.fine_width=768
            self.radius=16  

        pair_txt_path = os.path.join(self.root, opt.image_pairs_txt)
        if mode == 'train' and 'train' in opt.image_pairs_txt:
            self.mode = 'train'
        else:
            self.mode = 'test'
        with open(pair_txt_path, 'r') as f:
            lines = f.readlines()

        self.P_paths = []
        self.C_paths = []
        self.C_types = []
        for line in lines:
            p_name, c_name, c_type = line.strip().split()
            P_path = os.path.join(self.root, c_type, 'image', p_name)
            C_path = os.path.join(self.root, c_type, 'cloth_align', c_name)
            if self.resolution == 1024:
                P_path = P_path.replace('.png', '.jpg')
            self.P_paths.append(P_path)
            self.C_paths.append(C_path)
            self.C_types.append(c_type)

        ratio_dict = None
        if self.mode == 'train':
            ratio_dict = {}
            person_clothes_ratio_txt = os.path.join(self.root, 'person_clothes_ratio_upper_train.txt')
            with open(person_clothes_ratio_txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                c_name, ratio = line.strip().split()
                ratio = float(ratio)
                ratio_dict[c_name] = ratio
        self.ratio_dict = ratio_dict
        self.dataset_size = len(self.P_paths)

    ############### get palm mask ################
    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[..., np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps = [x1, y1, x2, y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / \
            (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / \
            (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1 < cos2:
            kps.extend([x3, y3, x4, y4])
        else:
            kps.extend([x4, y4, x3, y3])

        kps = np.array(kps).reshape(1, -1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask

    def get_hand_mask(self, hand_keypoints, h, w):
        # shoulder, elbow, wrist
        s_x, s_y, s_c = hand_keypoints[0]
        e_x, e_y, e_c = hand_keypoints[1]
        w_x, w_y, w_c = hand_keypoints[2]

        up_mask = np.ones((h, w, 1), dtype=np.float32)
        bottom_mask = np.ones((h, w, 1), dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            if self.resolution == 512:
                kernel = np.ones((50, 50), np.uint8)
            else:
                kernel = np.ones((100, 100), np.uint8)
            up_mask = cv2.dilate(up_mask, kernel, iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[..., np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            if self.resolution == 512:
                kernel = np.ones((30, 30), np.uint8)
            else:
                kernel = np.ones((60, 60), np.uint8)
            bottom_mask = cv2.dilate(bottom_mask, kernel, iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[..., np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)
                             == 2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, parsing, keypoints):
        h, w = parsing.shape[0:2]

        left_hand_keypoints = keypoints[[5, 6, 7], :].copy()
        right_hand_keypoints = keypoints[[2, 3, 4], :].copy()

        left_hand_up_mask, left_hand_bottom_mask = self.get_hand_mask(
            left_hand_keypoints, h, w)
        right_hand_up_mask, right_hand_bottom_mask = self.get_hand_mask(
            right_hand_keypoints, h, w)

        # mask refined by parsing
        left_hand_mask = (parsing == 15).astype(np.float32)
        right_hand_mask = (parsing == 16).astype(np.float32)

        left_palm_mask = self.get_palm_mask(
            left_hand_mask, left_hand_up_mask, left_hand_bottom_mask)
        right_palm_mask = self.get_palm_mask(
            right_hand_mask, right_hand_up_mask, right_hand_bottom_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ############### get palm mask ################

    def __getitem__(self, index):
        C_type = self.C_types[index]

        # person image
        P_path = self.P_paths[index]
        P = Image.open(P_path).convert('RGB')
        P_np = np.array(P)
        params = get_params(self.opt, P.size)
        transform_for_rgb = get_transform(self.opt, params)
        P_tensor = transform_for_rgb(P)

        # person 2d pose
        pose_path = P_path.replace('/image/', '/pose_25/')+'.npy'
        pose_data = np.load(pose_path)[0]
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx + r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = transform_for_rgb(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        Pose_tensor = pose_map

        # person 3d pose
        densepose_path = P_path.replace('/image/', '/densepose/')[:-4]+'.png'
        dense_mask = Image.open(densepose_path).convert('L')
        transform_for_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        dense_mask_tensor = transform_for_mask(dense_mask) * 255.0
        dense_mask_tensor = dense_mask_tensor[0:1, ...]

        # person parsing
        parsing_path = P_path.replace('/image/', '/parse-bytedance/')[:-4]+'.png'
        parsing = Image.open(parsing_path).convert('L')
        parsing_tensor = transform_for_mask(parsing) * 255.0

        parsing_np = (parsing_tensor.numpy().transpose(1, 2, 0)[..., 0:1]).astype(np.uint8)
        palm_mask_np = self.get_palm(parsing_np, pose_data)

        person_clothes_left_sleeve_mask_np = np.zeros_like(parsing_np)
        person_clothes_torso_mask_np = np.zeros_like(parsing_np)
        person_clothes_right_sleeve_mask_np = np.zeros_like(parsing_np)
        person_clothes_left_pants_mask_np = np.zeros_like(parsing_np)
        person_clothes_right_pants_mask_np = np.zeros_like(parsing_np)
        person_clothes_skirts_mask_np = np.zeros_like(parsing_np)
        neck_mask_np = np.zeros_like(parsing_np)
        left_hand_mask_np = np.zeros_like(parsing_np)
        right_hand_mask_np = np.zeros_like(parsing_np)
        hand_mask_np = np.zeros_like(parsing_np)

        if C_type == 'upper' or C_type == 'dresses':
            person_clothes_left_sleeve_mask_np = (parsing_np==21).astype(int) + \
                                                (parsing_np==24).astype(int)
            person_clothes_torso_mask_np = (parsing_np==5).astype(int) + \
                                        (parsing_np==6).astype(int)
            person_clothes_right_sleeve_mask_np = (parsing_np==22).astype(int) + \
                                                (parsing_np==25).astype(int)
            person_clothes_mask_np = person_clothes_left_sleeve_mask_np + \
                                  person_clothes_torso_mask_np + \
                                  person_clothes_right_sleeve_mask_np
            left_hand_mask_np = (parsing_np==15).astype(int)
            right_hand_mask_np = (parsing_np==16).astype(int)
            hand_mask_np = left_hand_mask_np + right_hand_mask_np
            neck_mask_np = (parsing_np==11).astype(int)
        else:
            person_clothes_left_pants_mask_np = (parsing_np==9).astype(int)
            person_clothes_right_pants_mask_np = (parsing_np==10).astype(int)
            person_clothes_skirts_mask_np = (parsing_np==13).astype(int)
            person_clothes_mask_np = person_clothes_left_pants_mask_np + \
                                  person_clothes_right_pants_mask_np + \
                                  person_clothes_skirts_mask_np

        person_clothes_mask_tensor = torch.tensor(person_clothes_mask_np.transpose(2, 0, 1)).float()
        person_clothes_left_sleeve_mask_tensor = torch.tensor(person_clothes_left_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_torso_mask_tensor = torch.tensor(person_clothes_torso_mask_np.transpose(2, 0, 1)).float()
        person_clothes_right_sleeve_mask_tensor = torch.tensor(person_clothes_right_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_left_pants_mask_tensor =  torch.tensor(person_clothes_left_pants_mask_np.transpose(2, 0, 1)).float()
        person_clothes_skirts_mask_tensor =  torch.tensor(person_clothes_skirts_mask_np.transpose(2, 0, 1)).float()
        person_clothes_right_pants_mask_tensor =  torch.tensor(person_clothes_right_pants_mask_np.transpose(2, 0, 1)).float()
        left_hand_mask_tensor = torch.tensor(left_hand_mask_np.transpose(2, 0, 1)).float()
        right_hand_mask_tensor = torch.tensor(right_hand_mask_np.transpose(2, 0, 1)).float()
        neck_mask_tensor = torch.tensor(neck_mask_np.transpose(2, 0, 1)).float()

        seg_gt_tensor = person_clothes_left_sleeve_mask_tensor * 1 + person_clothes_torso_mask_tensor * 2 + \
                        person_clothes_right_sleeve_mask_tensor * 3 +  person_clothes_left_pants_mask_tensor * 4 + \
                        person_clothes_skirts_mask_tensor * 5 + person_clothes_right_pants_mask_tensor * 6 + \
                        left_hand_mask_tensor * 7 + right_hand_mask_tensor * 8 + neck_mask_tensor * 9
        background_mask_tensor = 1 - (person_clothes_left_sleeve_mask_tensor + person_clothes_torso_mask_tensor + \
                                      person_clothes_right_sleeve_mask_tensor + person_clothes_left_pants_mask_tensor + \
                                      person_clothes_right_pants_mask_tensor + person_clothes_skirts_mask_tensor + \
                                      left_hand_mask_tensor + right_hand_mask_tensor + neck_mask_tensor)
        seg_gt_onehot_tensor = torch.cat([background_mask_tensor, person_clothes_left_sleeve_mask_tensor, \
                                         person_clothes_torso_mask_tensor, person_clothes_right_sleeve_mask_tensor, \
                                         person_clothes_left_pants_mask_tensor, person_clothes_skirts_mask_tensor, \
                                         person_clothes_right_pants_mask_tensor,  left_hand_mask_tensor, \
                                         right_hand_mask_tensor, neck_mask_tensor],0)

        if C_type == 'upper' or C_type == 'dresses':
            person_clothes_left_mask_tensor = person_clothes_left_sleeve_mask_tensor
            person_clothes_middle_mask_tensor = person_clothes_torso_mask_tensor
            person_clothes_right_mask_tensor = person_clothes_right_sleeve_mask_tensor
        else:
            person_clothes_left_mask_tensor = person_clothes_left_pants_mask_tensor
            person_clothes_middle_mask_tensor = person_clothes_skirts_mask_tensor
            person_clothes_right_mask_tensor = person_clothes_right_pants_mask_tensor

        ### preserve region mask
        ### preserve_mask1_np and preserve_mask2_np are only used for the training of warping module
        ### preserve_mask3_np is a bit different for the warping module and the try-on module
        if C_type == 'upper':
            if self.ratio_dict is None:
                preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
            else:
                pc_ratio = self.ratio_dict[self.C_paths[index].split('/')[-1][:-4]+'.png']
                if pc_ratio < 0.95:
                    preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
                elif pc_ratio < 1.0:
                    if random() < 0.5:
                        preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                    else:
                        preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,12,14,23,26,27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)
                else:
                    if random() < 0.1:
                        preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
                    else:
                        preserve_mask_for_loss_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,12,14,23,26,27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np,axis=0)

            preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,7,8,9,10,12,13,14,17,18,19,20,23,26,27,28]])
            preserve_mask_np = np.sum(preserve_mask_np,axis=0)

            preserve_mask1_np = preserve_mask_for_loss_np + palm_mask_np
            preserve_mask2_np = preserve_mask_for_loss_np + hand_mask_np
            preserve_mask3_np = preserve_mask_np + palm_mask_np
        elif C_type == 'dresses':
            preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,12,14,23]])
            if self.stage == 'gen':
                preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,12,14,23,8,19,20]])
            preserve_mask_np = np.sum(preserve_mask_np,axis=0)
            preserve_mask_for_loss_np = preserve_mask_np

            preserve_mask1_np = preserve_mask_for_loss_np + palm_mask_np
            preserve_mask2_np = preserve_mask_for_loss_np + hand_mask_np
            preserve_mask3_np = preserve_mask_np + palm_mask_np
        else:
            preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,5,6,7,11,12,14,15,16,21,22,23,24,25,26,27,28]])
            if self.stage == 'gen':
                preserve_mask_np = np.array([(parsing_np==index).astype(int) for index in [1,2,3,4,5,6,7,11,12,14,15,16,21,22,23,24,25,26,27,28,8,19,20]])
            preserve_mask_for_loss_np = preserve_mask_np

            preserve_mask1_np = np.sum(preserve_mask_for_loss_np,axis=0)
            preserve_mask2_np = np.sum(preserve_mask_for_loss_np, axis=0)
            preserve_mask3_np = np.sum(preserve_mask_np,axis=0)

        preserve_mask1_tensor = torch.tensor(preserve_mask1_np.transpose(2,0,1)).float()
        preserve_mask2_tensor = torch.tensor(preserve_mask2_np.transpose(2,0,1)).float()
        preserve_mask3_tensor = torch.tensor(preserve_mask3_np.transpose(2,0,1)).float()

        ### used for gradient truncation during training
        preserve_legs_mask_np = np.zeros_like(parsing_np)
        preserve_left_pants_mask_np = np.zeros_like(parsing_np)
        preserve_right_pants_mask_np = np.zeros_like(parsing_np)

        pants_mask_np = (parsing_np==9).astype(np.uint8) + (parsing_np==10).astype(np.uint8)
        skirts_mask_np = (parsing_np==13).astype(np.uint8)
        if C_type == 'lower':
            if np.sum(skirts_mask_np) > np.sum(pants_mask_np):
                preserve_legs_mask_np = (parsing_np==17).astype(np.uint8) + (parsing_np==18).astype(np.uint8) + \
                                        (parsing_np==19).astype(np.uint8) + (parsing_np==20).astype(np.uint8)
            else:
                preserve_left_pants_mask_np = (parsing_np==9).astype(np.uint8)
                preserve_right_pants_mask_np = (parsing_np==10).astype(np.uint8)
        elif C_type == 'dresses':
            preserve_legs_mask_np = (parsing_np==17).astype(np.uint8) + (parsing_np==18).astype(np.uint8) + \
                                    (parsing_np==19).astype(np.uint8) + (parsing_np==20).astype(np.uint8)
        
        preserve_legs_mask_tensor = torch.tensor(preserve_legs_mask_np.transpose(2,0,1)).float()
        preserve_left_pants_mask_tensor = torch.tensor(preserve_left_pants_mask_np.transpose(2,0,1)).float()
        preserve_right_pants_mask_tensor = torch.tensor(preserve_right_pants_mask_np.transpose(2,0,1)).float()
        

        ### clothes
        C_path = self.C_paths[index]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_for_rgb(C)

        CM_path = C_path.replace('/cloth_align/', '/cloth_align_mask-bytedance/')
        CM = Image.open(CM_path).convert('L')
        CM_tensor = transform_for_mask(CM)

        cloth_parsing_path = C_path.replace('/cloth_align/', '/cloth_align_parse-bytedance/')
        cloth_parsing = Image.open(cloth_parsing_path).convert('L')
        cloth_parsing_tensor = transform_for_mask(cloth_parsing) * 255.0
        cloth_parsing_tensor = cloth_parsing_tensor[0:1, ...]

        cloth_parsing_np = (cloth_parsing_tensor.numpy().transpose(1,2,0)).astype(int)
        if C_type == 'upper' or C_type == 'dresses':
            flat_clothes_left_mask_np = (cloth_parsing_np==21).astype(int)
            flat_clothes_middle_mask_np = (cloth_parsing_np==5).astype(int) + \
                                          (cloth_parsing_np==24).astype(int) + \
                                          (cloth_parsing_np==13).astype(int)
            flat_clothes_right_mask_np = (cloth_parsing_np==22).astype(int)
            flat_clothes_label_np = flat_clothes_left_mask_np * 1 + flat_clothes_middle_mask_np * 2 + flat_clothes_right_mask_np * 3
        else:
            flat_clothes_left_mask_np = (cloth_parsing_np==9).astype(int)
            flat_clothes_middle_mask_np = (cloth_parsing_np==13).astype(int)
            flat_clothes_right_mask_np = (cloth_parsing_np==10).astype(int)
            flat_clothes_label_np = flat_clothes_left_mask_np * 4 + flat_clothes_middle_mask_np * 5 + flat_clothes_right_mask_np * 6
        flat_clothes_label_np = flat_clothes_label_np / 6

        cloth_type_np = np.zeros_like(parsing_np)
        if C_type == 'upper':
            cloth_type_np = cloth_type_np + 1.0
        elif C_type == 'lower':
            cloth_type_np = cloth_type_np + 2.0
        else:
            cloth_type_np = cloth_type_np + 3.0
        cloth_type_np = cloth_type_np / 3.0
        
        flat_clothes_left_mask_tensor = torch.tensor(flat_clothes_left_mask_np.transpose(2, 0, 1)).float()
        flat_clothes_middle_mask_tensor = torch.tensor(flat_clothes_middle_mask_np.transpose(2, 0, 1)).float()
        flat_clothes_right_mask_tensor = torch.tensor(flat_clothes_right_mask_np.transpose(2, 0, 1)).float()

        flat_clothes_label_tensor = torch.tensor(flat_clothes_label_np.transpose(2, 0, 1)).float()
        cloth_type_tensor = torch.tensor(cloth_type_np.transpose(2,0,1)).float()

        WC_tensor = None
        WE_tensor = None
        AMC_tensor = None
        ANL_tensor = None
        if self.warproot:
            ### skin color
            face_mask_np = (parsing_np==14).astype(np.uint8)
            neck_mask_np = (parsing_np==11).astype(np.uint8)
            hand_mask_np = (parsing_np==15).astype(np.uint8) + (parsing_np==16).astype(np.uint8)
            leg_mask_np = (parsing_np==17).astype(int) + (parsing_np==18).astype(int)
            skin_mask_np = (face_mask_np+hand_mask_np+neck_mask_np+leg_mask_np)
            skin = skin_mask_np * P_np
            skin_r = skin[..., 0].reshape((-1))
            skin_g = skin[..., 1].reshape((-1))
            skin_b = skin[..., 2].reshape((-1))
            skin_r_valid_index = np.where(skin_r > 0)[0]
            skin_g_valid_index = np.where(skin_g > 0)[0]
            skin_b_valid_index = np.where(skin_b > 0)[0]

            skin_r_median = np.median(skin_r[skin_r_valid_index])
            skin_g_median = np.median( skin_g[skin_g_valid_index])
            skin_b_median = np.median(skin_b[skin_b_valid_index])

            arms_r = np.ones_like(parsing_np[...,0:1]) * skin_r_median
            arms_g = np.ones_like(parsing_np[...,0:1]) * skin_g_median
            arms_b = np.ones_like(parsing_np[...,0:1]) * skin_b_median
            arms_color = np.concatenate([arms_r,arms_g,arms_b],2).transpose(2,0,1)
            AMC_tensor = torch.FloatTensor(arms_color)
            AMC_tensor = AMC_tensor / 127.5 - 1.0

            # warped clothes
            warped_name = C_type + '___' + P_path.split('/')[-1] + '___' + C_path.split('/')[-1][:-4]+'.png'
            warped_path = os.path.join(self.warproot, warped_name)
            warped_result = Image.open(warped_path).convert('RGB')
            warped_result_np = np.array(warped_result)

            if self.resolution == 512:
                w = 384
            else:
                w = 768
            warped_cloth_np = warped_result_np[:,-2*w:-w,:]
            warped_parse_np = warped_result_np[:,-w:,:]

            warped_cloth = Image.fromarray(warped_cloth_np).convert('RGB')
            WC_tensor = transform_for_rgb(warped_cloth)

            warped_edge_np = (warped_parse_np==1).astype(np.uint8) + \
                             (warped_parse_np==2).astype(np.uint8) + \
                             (warped_parse_np==3).astype(np.uint8) + \
                             (warped_parse_np==4).astype(np.uint8) + \
                             (warped_parse_np==5).astype(np.uint8) + \
                             (warped_parse_np==6).astype(np.uint8)
            warped_edge = Image.fromarray(warped_edge_np).convert('L')
            WE_tensor = transform_for_mask(warped_edge) * 255.0
            WE_tensor = WE_tensor[0:1,...]
            preserve_mask3_tensor = preserve_mask3_tensor * (1-WE_tensor)

            arms_neck_label = (warped_parse_np==7).astype(np.uint8) * 1 + \
                              (warped_parse_np==8).astype(np.uint8) * 2 + \
                              (warped_parse_np==9).astype(np.uint8) * 3

            arms_neck_label = Image.fromarray(arms_neck_label).convert('L')
            ANL_tensor = transform_for_mask(arms_neck_label) * 255.0 / 3.0
            ANL_tensor = ANL_tensor[0:1,...]

        input_dict = {
            'image': P_tensor, 'pose':Pose_tensor , 'densepose':dense_mask_tensor,
            'seg_gt': seg_gt_tensor, 'seg_gt_onehot': seg_gt_onehot_tensor,
            'person_clothes_mask': person_clothes_mask_tensor,
            'person_clothes_left_mask': person_clothes_left_mask_tensor,
            'person_clothes_middle_mask': person_clothes_middle_mask_tensor,
            'person_clothes_right_mask': person_clothes_right_mask_tensor,
            'preserve_mask': preserve_mask1_tensor, 'preserve_mask2': preserve_mask2_tensor,
            'preserve_mask3': preserve_mask3_tensor,
            'color': C_tensor, 'edge': CM_tensor, 
            'flat_clothes_left_mask': flat_clothes_left_mask_tensor,
            'flat_clothes_middle_mask': flat_clothes_middle_mask_tensor,
            'flat_clothes_right_mask': flat_clothes_right_mask_tensor,
            'flat_clothes_label': flat_clothes_label_tensor,
            'flat_clothes_type': cloth_type_tensor,
            'c_type': C_type, 
            'color_path': C_path,
            'img_path': P_path,
            'preserve_legs_mask': preserve_legs_mask_tensor,
            'preserve_left_pants_mask': preserve_left_pants_mask_tensor,
            'preserve_right_pants_mask': preserve_right_pants_mask_tensor,
        }
        if WC_tensor is not None:
            input_dict['warped_cloth'] = WC_tensor
            input_dict['warped_edge'] = WE_tensor
            input_dict['arms_color'] = AMC_tensor
            input_dict['arms_neck_lable'] = ANL_tensor

        return input_dict

    def __len__(self):
        if self.mode == 'train':
            return len(self.P_paths) // (self.opt.batchSize * self.opt.num_gpus) * (self.opt.batchSize * self.opt.num_gpus)
        else:
            return len(self.P_paths)

    def name(self):
        return 'AlignedDataset'
