'''
这个脚本把swap_face封装为一个类，可以直接调用。
这是加了att后的模型，代码保留了原版参数，调整if语句可以恢复原版。同时包含光照的调节实验代码，已注释。

@date: 2020.9.14
@author: zhuzhou
'''


import time
import yaml
import face_alignment
import os,sys
import subprocess
from data_process.align_images import *

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte

import torch
from sync_batchnorm import DataParallelWithCallback
import torch.nn.functional as F

from modules.segmentation_module import SegmentationModule
from modules.reconstruction_module import ReconstructionModule
from logger import load_reconstruction_module, load_segmentation_module

from modules.util import AntiAliasInterpolation2d
from modules.dense_motion import DenseMotionNetwork

import warnings

warnings.filterwarnings('ignore')
if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

# 固定所有随机数
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class PartSwapGenerator(ReconstructionModule):
    def __init__(self, blend_scale=1, first_order_motion_model=False, remain_ratio=1., dense_config=None, **kwargs):
        super(PartSwapGenerator, self).__init__(**kwargs)
        if blend_scale == 1:
            self.blend_downsample = lambda x: x
        else:
            self.blend_downsample = AntiAliasInterpolation2d(1, blend_scale)
            self.remain_ratio = remain_ratio  # How much to stay with source image

        if first_order_motion_model:
            self.dense_motion_network = DenseMotionNetwork(**dense_config)
        else:
            self.dense_motion_network = None

    def forward(self, source_image, target_image, target_affine, target_shift, source_affine, source_shift, blend_mask, use_source_segmentation=False):
        '''
        image: 原图片
        shift: 提取出来的关键点，是一组xy的坐标
        affine: 各个关键点对应的一阶倒数，每个关键点对应两组xy的坐标
        '''
        # Encoding of source image
        enc_source = self.first(source_image)
        for i in range(len(self.down_blocks)):
            enc_source = self.down_blocks[i](enc_source)

        # Encoding of target image
        enc_target = self.first(target_image)
        for i in range(len(self.down_blocks)):
            enc_target = self.down_blocks[i](enc_target)

        # Compute flow field for source image
        motion = self.dense_motion_network(source_image, target_affine, source_affine, target_shift, source_shift)
        deformation = motion['deformation']

        # Deform source encoding according to the motion
        enc_source = self.deform_input(enc_source, deformation)

        if self.estimate_visibility:
            visibility = motion['visibility']

            if enc_source.shape[2] != visibility.shape[2] or enc_source.shape[3] != visibility.shape[3]:
                visibility = F.interpolate(visibility, size=enc_source.shape[2:], mode='bilinear')
            enc_source = enc_source * visibility

        blend_mask = self.blend_downsample(blend_mask)

        out = enc_target * (1 - blend_mask*self.remain_ratio) + enc_source * blend_mask*self.remain_ratio

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)

        out = self.final(out)
        out = torch.sigmoid(out)

        return out


class face_swaper:
    def __init__(self, checkpoint="first-order-512-best.pth.tar", batch_size=4, device_ids=[0,1,2,3]):
        self.batch_size = batch_size
        self.device_ids = device_ids
        # 用最后一张卡跑ffmpeg命令，这个ffmpeg必须是支持cuda的，如果不支持，后面的命令要改编码方式
        self.ffmpeg = f"CUDA_VISIBLE_DEVICES={self.device_ids[-1]} /usr/local/bin/ffmpeg"  # "/usr/bin/ffmpeg"

        # 初始化模型
        torch.cuda.set_device(device_ids[0])
        self.reconstruction_module, self.segmentation_module = self._load_checkpoints(checkpoint, 0.25, True, remain_ratio=1., device_ids=device_ids)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.face_parser = self._load_face_parser()

        # 默认: [1,2,3,4,5,6,7,8,9,10,11,12,13]
        # 用原视频的嘴: [1,2,3,4,5,6,7,8,9,10]  # 11, 12, 13跟嘴相关
        self.swap_index = [1,2,3,4,5,6,10,11,12,13]  # 789可能跟耳朵相关


    def _load_checkpoints(self, checkpoint, blend_scale=0.125, first_order_motion_model=False, remain_ratio=1., device_ids=[0]):
        #############################################
        # 模型参数
        #############################################
        if False:  # 源码模型参数
            common_config = {'num_segments': 10,
                            'num_channels': 3,
                            'estimate_affine_part': True}
            dense_config = {'block_expansion': 64,
                            'max_features': 1024,
                            'num_blocks': 6,
                            'scale_factor': 0.25}
            recon_config = {'block_expansion': 64,
                            'max_features': 512,
                            'num_down_blocks': 2,
                            'num_bottleneck_blocks': 9,
                            'estimate_visibility': True}
            seg_config = {'temperature': 0.1,
                        'block_expansion': 32,
                        'max_features': 1024,
                        'scale_factor': 0.25,
                        'num_blocks': 5}
        else:  # 加了att参数
            common_config = {'num_segments': 10,
                            'num_channels': 3,
                            'estimate_affine_part': True}
            dense_config = {'block_expansion': 64,
                            'max_features': 1024,
                            'num_blocks': 5,
                            'scale_factor': 0.25,
                            'use_att': True}
            recon_config = {'block_expansion': 64,
                            'max_features': 512,
                            'num_down_blocks': 2,
                            'num_bottleneck_blocks': 9,
                            'estimate_visibility': True}
            seg_config = {'temperature': 0.1,
                        'block_expansion': 32,
                        'max_features': 1024,
                        'scale_factor': 0.25,
                        'num_blocks': 5}
        #############################################

        reconstruction_module = PartSwapGenerator(blend_scale=blend_scale,
                                                first_order_motion_model=first_order_motion_model,
                                                remain_ratio=remain_ratio,
                                                dense_config=dense_config,
                                                **recon_config,
                                                **common_config)
        reconstruction_module.cuda()
        segmentation_module = SegmentationModule(**seg_config,
                                                **common_config)
        segmentation_module.cuda()
        checkpoint = torch.load(checkpoint)

        gen_keys = checkpoint['generator'].keys() if 'generator' in checkpoint else checkpoint['reconstruction_module'].keys()
        if 'blend_downsample.weight' in gen_keys:
            print("'blend_downsample.weight' in gen_keys")

        load_reconstruction_module(reconstruction_module, checkpoint)
        load_segmentation_module(segmentation_module, checkpoint)

        reconstruction_module = DataParallelWithCallback(reconstruction_module, device_ids=device_ids)
        segmentation_module = DataParallelWithCallback(segmentation_module, device_ids=device_ids)

        reconstruction_module.eval()
        segmentation_module.eval()

        return reconstruction_module, segmentation_module


    def _load_face_parser(self):
        from face_parsing.model import BiSeNet

        face_parser = BiSeNet(n_classes=19)
        face_parser.cuda()
        face_parser.load_state_dict(torch.load('face_parsing/cp/79999_iter.pth'))
        face_parser.eval()

        mean = torch.Tensor(np.array([0.485, 0.456, 0.406], dtype=np.float32)).view(1, 3, 1, 1)
        std = torch.Tensor(np.array([0.229, 0.224, 0.225], dtype=np.float32)).view(1, 3, 1, 1)

        face_parser.mean = mean.cuda()
        face_parser.std = std.cuda()

        return face_parser


    def _extract_bbox(self, frame):
        if max(frame.shape[0], frame.shape[1]) > 640:
            scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
            frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
            frame = img_as_ubyte(frame)
        else:
            scale_factor = 1
        frame = frame[..., :3]
        bboxes = self.fa.face_detector.detect_from_image(frame[..., ::-1])

        if len(bboxes) == 0:
            return []
        return np.array(bboxes)[:, :-1] * scale_factor


    def _process_video_stable(self, inp, image_shape, tmp_folder, increase_area=0.1):
        '''
        假设视频中人脸运动不大的情况下可以认为第一帧就提供了人脸位置信息，然后整个视频中人脸基本都在该区域。
        '''
        video = imageio.get_reader(inp)
        fps = video.get_meta_data()['fps']
        end = video.count_frames()

        for i, frame in enumerate(video):
            try:
                frame_shape = frame.shape
                bboxes = self._extract_bbox(frame)
                bbox = bboxes[0]
            except:
                continue
            start = i
            break

        # Compute w, h, left, top
        left, top, right, bot = bbox
        width = right - left
        height = bot - top

        #Computing aspect preserving bbox
        width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
        height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

        left = int(left - width_increase * width)
        top = int(top - height_increase * height)
        right = int(right + width_increase * width)
        bot = int(bot + height_increase * height)

        top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
        h, w = bot - top, right - left

        start = start/fps
        time = (end-start)/fps

        command = f'{self.ffmpeg} -y -i {inp} -vf "crop={w}:{h}:{left}:{top}, scale={image_shape[0]}:{image_shape[1]}" -ss {start} -t {time} -c:v h264_nvenc -c:a copy -v quiet crop.mp4'

        # 如果第一帧没有人脸，直接在原视频上把没脸的视频部分切了
        if start:
            new_target_video = tmp_folder+'/'+inp.split('/')[-1]
            p = subprocess.Popen(f"{self.ffmpeg} -y -i {inp} -ss {start} -t {time} -c:v h264_nvenc -v quiet {new_target_video}", shell=True)
            p.wait()
            inp = new_target_video
        return command, inp


    def _make_video(self, swap_index, source_image, target_video, reconstruction_module, segmentation_module, bz=2, face_parser=None):
        assert type(swap_index) == list
        use_source_segmentation = False  # 强行将其设置为False，方便之后的source和target固定
        is_hard = False
        use_color_old_method = True  # True就是用之前的颜色分布算法，False就是低通滤波加高通滤波的光照学习

        if not use_color_old_method:
            def get_kernel(scale, low=True):
                channels = 3
                sigma = (1 / scale - 1) / 2
                kernel_size = 2 * round(sigma * 4) + 1
                ka = kernel_size // 2
                kb = ka - 1 if kernel_size % 2 == 0 else ka
                kernel_size = [kernel_size, kernel_size]
                sigma = [sigma, sigma]
                kernel = 1
                meshgrids = torch.meshgrid(
                    [
                        torch.arange(size, dtype=torch.float32)
                        for size in kernel_size
                    ]
                )
                for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                    mean = (size - 1) / 2
                    if low:
                        kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))
                    else:
                        kernel *= (1-torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2)))

                # Make sure sum of values in gaussian kernel equals 1.
                kernel = kernel / torch.sum(kernel)
                # Reshape to depthwise convolutional weight
                kernel = kernel.view(1, 1, *kernel.size())
                kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
                kernel = kernel.cuda()
                return kernel, ka, kb


        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            source = source.cuda()
            # target = torch.tensor(np.array(target_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            target = torch.tensor(np.array(target_video).astype(np.float32)).permute(0, 3, 1, 2)

            seg_shift, seg_affine = segmentation_module(source)

            ################################################################
            # 这里是计算source的脸部区域，并做mean和std的计算
            ################################################################
            parser_mean = face_parser.mean
            parser_std = face_parser.std
            if face_parser is not None:
                source_mask = (source-face_parser.mean)/face_parser.std
                source_mask = torch.softmax(face_parser(source_mask)[0], dim=1)
            else:
                raise Exception
            # 这里计算平均值的时候用的是source的脸、鼻子、眉毛、上嘴唇、下嘴唇、脖子7个部分计算的，因为之后在做分布转化的时候如果没有眉毛部分，由于分割算法不精细会有镂空的现象
            source_mask_face = source_mask[:, [1,2,3,10,12,13,14]].sum(dim=1, keepdim=True)
            source_mask_face = (source_mask_face>0.5).view(512, 512)
            # 只求脸和鼻子部分的mean和std
            source_mask_face2 = source_mask[:, [1,10,14]].sum(dim=1, keepdim=True)
            # source_mask_face2 = (source_mask_face2>0.5).view(512, 512).type(source_mask_face2.type())
            source_mask = (source*source_mask_face2).view(1,3,-1)
            source_mean = torch.mean(source_mask, dim=-1).squeeze()
            source_std = torch.std(source_mask, dim=-1).squeeze()
            #################################################################

            ## 批量计算seg_targets
            n = target.shape[0]
            shifts = []
            affines = []

            start = time.time()
            for batch_i in range((n + bz - 1) // bz):
                batch_t = target[(batch_i*bz):(batch_i*bz+bz)].cuda()
                shift, affine = segmentation_module(batch_t)
                shifts.extend(shift)
                affines.extend(affine)
            shifts = torch.stack(shifts, dim=0)
            affines = torch.stack(affines, dim=0)
            print("      - 脸部动作计算时间: ", time.time()-start)

            ## 逐帧计算target frame的脸部分割信息
            source_images = []
            blend_mask_parts = []
            start = time.time()
            for frame_idx in range(n):
                target_frame = target[frame_idx].unsqueeze(dim=0).cuda()

                # 计算当前target frame的脸部分割信息
                blend_mask = (target_frame - parser_mean) / parser_std
                blend_mask = torch.softmax(face_parser(blend_mask)[0], dim=1)
                blend_mask_part = blend_mask[:, swap_index].sum(dim=1, keepdim=True)
                blend_mask_face = blend_mask[:, [1,10,14]].sum(dim=1, keepdim=True)
                if is_hard:
                    # blend_mask_part[blend_mask_part<0.4] = 0.
                    blend_mask_part = (blend_mask_part>0.5).type(blend_mask_part.type())
                blend_mask_parts.extend(blend_mask_part)

                if use_color_old_method:
                    ################################################################
                    # 对target图片做色彩的mean和std计算，只做脸部
                    # 最简单的学习色彩值分布规律
                    ################################################################
                    target_mask = (target_frame*blend_mask_face).view(1,3,-1)
                    target_mean = torch.mean(target_mask, dim=-1).squeeze()
                    target_std = torch.std(target_mask, dim=-1).squeeze()

                    # 把source的分布变为target的分布
                    diff_mean = source_mean-target_mean
                    diff_std = source_std/target_std
                    diff_mean *= 0.5

                    source_cp = source[0].clone()
                    # for c in range(3):
                    #     source_cp[c][source_mask_face] = (source_cp[c][source_mask_face]-diff_mean[c])/diff_std[c]
                    #     source_cp = torch.clamp(source_cp, min=0., max=1.)
                else:
                    source_cp = source[0].clone()
                source_images.append(source_cp)
                ################################################################
            print("      - 脸部分割时间: ", time.time()-start)
            source_images = torch.stack(source_images, dim=0)
            blend_mask_parts = torch.stack(blend_mask_parts, dim=0)

            source_affine = seg_affine.repeat(n,1,1,1).view(n,10,2,2)
            source_shift = seg_shift.repeat(n,1,1).view(n,10,2)

            ## 批量计算生成图片
            predictions = []
            start = time.time()
            for batch_i in range((n + bz - 1) // bz):
                bt_source = source_images[batch_i*bz:(batch_i*bz+bz)]
                bt_target = target[batch_i*bz:(batch_i*bz+bz)]
                bt_tgt_affine = affines[batch_i*bz:(batch_i*bz+bz)]
                bt_tgt_shift = shifts[batch_i*bz:(batch_i*bz+bz)]
                bt_src_affine = source_affine[batch_i*bz:(batch_i*bz+bz)]
                bt_src_shift = source_shift[batch_i*bz:(batch_i*bz+bz)]
                bt_blend_mask = blend_mask_parts[batch_i*bz:(batch_i*bz+bz)]
                out = reconstruction_module(source_image=bt_source, target_image=bt_target, target_affine=bt_tgt_affine, target_shift=bt_tgt_shift, source_affine=bt_src_affine, source_shift=bt_src_shift, blend_mask = bt_blend_mask)


                #####
                # 尝试新的光照迁移方法，用原来的视频帧里的光照信息加到生成后的视频上
                #####
                if not use_color_old_method:
                    low_kernel,ka,kb = get_kernel(0.0625)

                    # low_out = F.pad(out, (ka, kb, ka, kb))
                    # low_out = F.conv2d(low_out, weight=high_kernel, groups=3)
                    # high_out = out - low_out
                    bt_target = bt_target.cuda()
                    bt_target = F.pad(bt_target, (ka, kb, ka, kb))
                    low_in = F.conv2d(bt_target, weight=low_kernel, groups=3)

                    bt_blend_mask_hard = (bt_blend_mask > 0.5).repeat(1,3,1,1)

                     ### Save image
                    from modules.util import save_img

                    save_img(out, 'out.jpg')
                    # save_img(low_out, 'low_out.jpg')
                    save_img(high_out, 'high_out.jpg')
                    save_img(low_in, 'low_in.jpg')

                    out[bt_blend_mask_hard] = 0.5*out[bt_blend_mask_hard] + 0.5*low_in[bt_blend_mask_hard]

                predictions.extend(np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1]))

        print("      - 生成图片时间: ", time.time()-start)
        return predictions


    def swap_face(self, source_image, target_video, tmp_folder, result_video_name, verbose=True):
        # 生成中间文件存放文件夹
        os.makedirs(tmp_folder, exist_ok=True)

        if verbose:
            print("="*30)
            print(" Source image: ", source_image, "\n Target video: ", target_video)

        all_start = time.time()

        ######################################
        ## 1. 视频裁剪和对齐
        ######################################
        start = time.time()
        # 视频裁剪
        target_video_crop = tmp_folder+"/"+target_video.split("/")[-1][:-4]+"_crop.mp4"
        crop_info_txt = tmp_folder+"/"+target_video.split("/")[-1][:-4]+"_crop_info.txt"
        if not os.path.exists(target_video_crop) or not os.path.exists(crop_info_txt):
            command, target_video = self._process_video_stable(target_video, (512,512), tmp_folder, increase_area=0.2)  # 脸部动作比较大的时候increase_area相应增大
            crop_info = command.split('"')[1].split(",")[0][5:].split(":")
            with open(crop_info_txt, "w") as fout:
                fout.write(",".join(crop_info))
            command = command.replace("crop.mp4", target_video_crop)
            # if verbose:
            #     print(command)
            p = subprocess.Popen(command, shell=True)
            p.wait()
            crop_info = list(map(int, crop_info))
        else:
            with open(crop_info_txt, "r") as fin:
                for l in fin:
                    crop_info = list(map(int, l.split(",")))
                    break
        #　图片对齐
        raw_img_path = source_image
        ali_img_path = tmp_folder+"/"+raw_img_path.split("/")[-1].split(".")[0]+'_ali.jpg'
        # If already do align, pass it
        if verbose:
            print('ali_img_path: ', ali_img_path)
        if not os.path.exists(ali_img_path):
            # 获取crop视频的第一帧
            crop_first_frame = f"{target_video_crop[:-4]}_frame1.jpg"
            p = subprocess.Popen(f"{self.ffmpeg} -y -i {target_video_crop} -frames 1 -v quiet {crop_first_frame}", shell=True)
            p.wait()
            print('Getting alined image ...')
            my_aligner = aligner(output_size=512)
            my_aligner.align(raw_img_path, ali_img_path, crop_first_frame)
            my_aligner.close()
            if not os.path.exists(ali_img_path):
                print('[ERROR] No aligned image found. Please check aligner.')
                raise Exception('[ERROR] No aligned image found. Please check aligner.')
        source_image = imageio.imread(ali_img_path)

        if verbose:
            print('   - 图片对齐时间: ', time.time()-start)

        ######################################
        ## 2. 读视频
        ######################################
        reader = imageio.get_reader(target_video_crop)
        fps = reader.get_meta_data()['fps']
        target_videos = []
        try:
            for im in reader:
                target_videos.append(im)
        except RuntimeError:
            pass
        reader.close()

        source_image = resize(source_image, (512, 512))[..., :3]
        target_videos = [resize(frame, (512, 512))[..., :3] for frame in target_videos]

        ######################################
        ## 3. 换脸
        ######################################
        start = time.time()
        predictions = self._make_video(self.swap_index, source_image, target_videos, self.reconstruction_module, self.segmentation_module, self.batch_size*len(self.device_ids), self.face_parser)

        imageio.mimsave(result_video_name, [img_as_ubyte(frame) for frame in predictions], fps=fps)
        if verbose:
            print('   - 换脸时间: ', time.time()-start, ', images num: ', len(predictions))

        ######################################
        ## 4. 拼接原视频
        ######################################
        start = time.time()
        out_file = result_video_name[:-4]+'_scale.mp4'
        p = subprocess.Popen(f"{self.ffmpeg} -y -resize {int(int(crop_info[0]/2)*2)}x{int(int(crop_info[1]/2)*2)} -c:v h264_cuvid -i {result_video_name} -b:v 8M -v quiet {out_file}", shell=True)

        p.wait()
        in_file = out_file
        out_file = result_video_name
        p = subprocess.Popen(f"{self.ffmpeg} -y -i {target_video} -i {in_file} -filter_complex \"overlay={crop_info[2]}:{crop_info[3]}\" -c:v h264_nvenc -b:v 8M -c:a copy -v quiet {out_file}", shell=True)
        p.wait()
        p = subprocess.Popen(f"{self.ffmpeg} -y -i {out_file} -i {target_video} -map 0 -c:v h264_nvenc -b:v 8M -map 1 -c:a copy -strict experimental -pix_fmt yuv420p -v quiet {out_file}", shell=True)
        p.wait()
        if verbose:
            print('   - 拼接至原视频时间: ', time.time()-start)
            print('   - 总体运行时间: ', time.time()-all_start)
            print(f"- Done, save to {out_file}")
            print("="*30+"\n")
