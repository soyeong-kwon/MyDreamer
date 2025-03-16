# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import os
import shutil
import pathlib
from PIL import Image, ImageFilter
from typing import AnyStr, Union, Tuple, List
import pandas as pd 

import cv2
import cairosvg
import omegaconf
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from torchvision import transforms
from skimage.color import rgb2gray

from mydreamer.libs import ModelState, get_optimizer
from mydreamer.painter import (CompPainter, CompPainterOptimizer, xing_loss_fn, Painter, PainterOptimizer,
                                CosineWithWarmupLRLambda, MySDSPipeline, LSDSPipeline, DiffusionPipeline)
from mydreamer.token2attn.attn_control import EmptyControl, AttentionStore
from mydreamer.token2attn.ptp_utils import view_images
from mydreamer.utils.plot import plot_img, plot_couple, plot_attn, save_image, plot_attn2
from mydreamer.utils import init_tensor_with_color, AnyPath, mkdir
from mydreamer.svgtools import merge_svg_files, is_valid_svg, split_svg
from mydreamer.diffusers_warp import model2res

## fft filter 
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from skimage import io
from scipy.ndimage import gaussian_filter
import math 
from torchvision.transforms import ToPILImage

import random 

class MyDreamerPipeline(ModelState):

    def __init__(self, args):
        # assert

        logdir_ = f"sd{args.seed}" \
                  f"-{'vpsd' if args.skip_sive else 'sive'}" \
                  f"-{args.x.style}" \
                  f"-P{args.x.num_paths}" \
                  f"{'-RePath' if args.x.path_reinit.use else ''}"
        super().__init__(args, log_path_suffix=logdir_)

        """FIVE log dirs"""
        self.five_attn_dir = self.result_path / "FIVE_attn_logs"
        self.five_freq_dir = self.result_path / "FIVE_freq_logs"
        self.five_logs_dir = self.result_path / "FIVE_iter_logs"
        self.five_final_dir = self.result_path / "FIVE_result_logs"
        
        """DVSD log dirs"""
        self.sd_bg_png_logs_dir = self.result_path / "SD_bg_png_logs"
        self.sd_bg_svg_logs_dir = self.result_path / "SD_bg_svg_logs"
        self.sd_obj_png_logs_dir = self.result_path / "SD_obj_png_logs"
        self.sd_obj_svg_logs_dir = self.result_path / "SD_obj_svg_logs"
        self.sd_png_logs_dir = self.result_path / "SD_png_logs"
        self.sd_svg_logs_dir = self.result_path / "SD_svg_logs"
        
        mkdir([self.five_attn_dir, self.five_freq_dir, self.five_logs_dir, self.five_final_dir,
               self.sd_bg_png_logs_dir, self.sd_bg_svg_logs_dir, self.sd_obj_png_logs_dir, self.sd_obj_svg_logs_dir,
               self.sd_png_logs_dir, self.sd_svg_logs_dir])

        # make video log
        self.make_video = self.args.mv
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = self.result_path / "frame_logs"
            self.frame_log_dir.mkdir(parents=True, exist_ok=True)

        # torch Generator seed
        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

        # for convenience
        self.style = self.x_cfg.style
        self.im_size = self.x_cfg.image_size
        self.sive_cfg = self.x_cfg.sive
        self.sive_optim = self.x_cfg.sive_stage_optim
        self.vpsd_cfg = self.x_cfg.vpsd
        self.vpsd_optim = self.x_cfg.vpsd_stage_optim
        
        ######### alpha, dynamic range 

        self.gs_range = args.run_type.gs_range

    def painterly_rendering(self, text_prompt: str, target_file: AnyPath = None):
        self.args.skip_sive = False
        # log prompts
        self.print(f"prompt: {text_prompt}")
        self.print(f"neg_prompt: {self.args.neg_prompt}\n")
        
        for i in range(self.num_samples):
            # mode 3: FIVE + VPSD
            input_svg_path, input_images = self.FIVE_stage(text_prompt, i)
            self.print("SVG fine-tuning via VPSD...")
            self.VPSD_stage(text_prompt, init_svg_path=input_svg_path, init_image=input_images)
            self.close(msg="painterly rendering complete.")

    def split_renderer(self, 
                       bg_svg_path, 
                       obj_svg_path, 
                       attn_map_path, 
                       tau):        
        
        dummy_renderer = self.load_renderer()
        dummy_renderer.component_wise_path_init(gt = None, pred=None, init_type=self.x_cfg.coord_init)
        
        # set mask 
        attn_map = Image.open(attn_map_path).convert("L")
        attn_map = torch.from_numpy(np.array(attn_map)).float() / 255.0
        
        bool_attn_map = attn_map > tau
        mask = bool_attn_map.int()  # [w, h]
    
        o_idx = 0; obj_shapes = []; obj_shape_groups =  []
        b_idx = 0; bg_shapes = []; bg_shape_groups = []

        for i in range(self.x_cfg.num_paths):
            path = self.merged_renderer.shapes[i]
            shape_group = self.merged_renderer.shape_groups[i]
            
            tensor_points = path.points
            center_points = tensor_points.mean(dim=0)
            center_x, center_y = int(center_points[0]), int(center_points[1])

            try:
                if mask[center_y, center_x] == 1:  
                    obj_shapes.append(path) 
                    shape_group.shape_ids = torch.tensor([o_idx])
                    obj_shape_groups.append(shape_group)
                    o_idx += 1
                else: 
                    bg_shapes.append(path)
                    shape_group.shape_ids = torch.tensor([b_idx])
                    bg_shape_groups.append(shape_group)
                    b_idx += 1
            except: 
                    print(center_y, center_x)
                    bg_shapes.append(path)
                    shape_group.shape_ids = torch.tensor([b_idx])
                    bg_shape_groups.append(shape_group)
                    b_idx += 1

        self.print(f"==> Object Renderer path : {len(obj_shape_groups)}")
        self.print(f"==> Background Renderer path : {len(bg_shape_groups)}")
        self.print(f"==> Total Renderer path : {len(obj_shape_groups) + len(bg_shape_groups)}")
    
        dummy_renderer.save_svg(filename= obj_svg_path, 
                            width=600, 
                            height=600,
                            shapes = obj_shapes,
                            shape_groups= obj_shape_groups)
        
        dummy_renderer.save_svg(filename= bg_svg_path, 
                            width=600, 
                            height=600,
                            shapes = bg_shapes,
                            shape_groups= bg_shape_groups)
        
        # svg2png 
        cairosvg.svg2png(url=bg_svg_path, write_to=bg_svg_path.replace('.svg', '.png'))
        cairosvg.svg2png(url=obj_svg_path, write_to=obj_svg_path.replace('.svg', '.png'))
        
        del dummy_renderer

    def SD_stage(self, 
                text_prompt: str, 
                prev_prompt: str, 
                batch_idx: int, 
                merged_svg_path: str,
                attn_path: str, 
                gt_path: str):
        
        # for convenience
        guidance_cfg = self.x_cfg.vpsd
        sd_model_cfg = self.x_cfg.vpsd_model_cfg
        n_particle = guidance_cfg.n_particle
        path_reinit = self.x_cfg.path_reinit
        
        #### sy ####
        total_step = guidance_cfg.num_iter
        gs_range = list(map(float, self.gs_range.split("_"))) 
        pipeline = MySDSPipeline.from_pretrained(self.x_cfg.vpsd_model_cfg.model_id,
                                        torch_dtype=self.weight_dtype, 
                                        local_files_only = False, 
                                        force_download=False,
                                        resume_download=False).to(self.device)

        pipeline.t_range = [ 0.02, 0.98 ]
        pipeline.t_schedule = 'randint'
        
        # pipeline에 class 변수 추가 
        pipeline.total_step = total_step
        pipeline.gs_min = torch.tensor(float(gs_range[0])) 
        pipeline.gs_max = torch.tensor(float(gs_range[1]))
        gs = pipeline.gs_max 

        # 기본이 되는 이미지 설정 
        gt_img = self.target_file_preprocess(gt_path)

        ########################
        #  object, background에 따라 Decompose 
        result_path = str(self.result_path)
        bg_svg_path = result_path + f"/{batch_idx}/SD_split_bg.svg"
        obj_svg_path = result_path + f"/{batch_idx}/SD_split_obj.svg"
        
        # initialize merged renderer

        merged_renderer = self.load_renderer(merged_svg_path)
        init_image = self.target_file_preprocess(str(merged_svg_path).replace('.svg', '.png'))
        merged_renderer.component_wise_path_init(gt=init_image, pred=None, init_type=self.x_cfg.coord_init)
        merged_renderer.init_image(num_paths=self.x_cfg.num_paths)
        
        self.merged_renderer = merged_renderer
        
        self.split_renderer(bg_svg_path = bg_svg_path, 
                       obj_svg_path= obj_svg_path, 
                       attn_map_path= attn_path, 
                       tau = 0.3)
    
        init_svg_path = [bg_svg_path, obj_svg_path]
        init_image = [self.target_file_preprocess(bg_svg_path.replace('.svg', '.png')), 
                      self.target_file_preprocess(obj_svg_path.replace('.svg', '.png'))]

        renderers = [self.load_renderer(init_path) for init_path in init_svg_path]
        for render, gt_ in zip(renderers, init_image):         # initialize the particles
            render.component_wise_path_init(gt=gt_, pred=None, init_type=self.x_cfg.coord_init)
        for t, r in enumerate(renderers):
            num_paths = len(r.shapes)
            init_imgs = r.init_image(num_paths=num_paths)
            plot_img(init_imgs, self.result_path, fname=f"init_img_stage_two_{t}")

        ########################

        optimizers = []
        for renderer in renderers:
            optim_ = PainterOptimizer(renderer,
                                      self.style,
                                      guidance_cfg.num_iter,
                                      self.vpsd_optim,
                                      self.x_cfg.trainable_bg)
            optim_.init_optimizers()
            optimizers.append(optim_)    
            
        self.print(f"-> Painter point Params: {len(renderers[0].get_point_parameters())}")
        self.print(f"-> Painter color Params: {len(renderers[0].get_color_parameters())}")
        self.print(f"-> Painter width Params: {len(renderers[0].get_width_parameters())}")

        self.step = 0  # reset global step
        self.print(f"Total Optimization Steps: {total_step}")
        
        ## SET PROMPT ## 
        negative_prompts = [self.args.neg_prompt]
        gen_prompts = [text_prompt]
        prev_prompts = [prev_prompt]
        
        print(gen_prompts, prev_prompts)
        self.print(self.flag_igs, self.gs_type)
        # mse_loss = torch.nn.MSELoss()
        # data = {"Step": [], "MSE": []}
        
        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_step:
               
                # set particles
                bg_tensor = renderers[0].render_warp() 
                obj_tensor = renderers[1].render_warp()
                obj_img = renderers[1].get_image()
                bg_img = renderers[0].get_image()
                
                # raster_img = self.get_combined_img(bg_tensor, obj_tensor)  # torch.Size([1, 3, 600, 600])
                raster_img = self.alpha_blend(bg_tensor, obj_tensor)  # torch.Size([1, 3, 600, 600])
                raster_img = raster_img.to(self.device)
                self.vis_image(raster_img, fpath = f'{batch_idx}/SD_merged_img.png')
                
                # decomposed optimization loss
                if self.step < 400 and self.flag_prev: 
                    L_merge = torch.tensor(0.)
                    gs_tmp = random.uniform(7.5, 15)
                    L_concept, grad, t_step = pipeline.sds_default(
                        pred_rgb = obj_img.to(self.weight_dtype),
                        im_size = model2res(sd_model_cfg.model_id),
                        prompts=prev_prompts,
                        negative_prompts=negative_prompts,
                        step = self.step,
                        grad_scale=guidance_cfg.grad_scale, 
                        guidance_scale_ = gs_tmp)
                        # guidance_scale_ = pipeline.gs_min)
                    L_bg, grad, t_step = pipeline.sds_default(
                        pred_rgb = bg_img.to(self.weight_dtype), 
                        im_size = model2res(sd_model_cfg.model_id), 
                        prompts = gen_prompts, 
                        negative_prompts = negative_prompts, 
                        step = self.step, 
                        grad_scale = guidance_cfg.grad_scale, 
                        guidance_scale_ = pipeline.gs_max, 
                        noise_step = t_step
                    )
                else: 
                    if self.flag_igs == True: # DGS
                        if self.gs_type == 'linear': # DGS_Linear
                            gs = pipeline.gs_min + (pipeline.gs_max - pipeline.gs_min) * (1 - (self.step / total_step))
                        elif self.gs_type == 'cosine': # DGS_Cosine
                            gs = pipeline.gs_min + (pipeline.gs_max - pipeline.gs_min) * 0.5 * (1 + math.cos(math.pi * (self.step / total_step))) 
                    else: 
                        if self.gs_type == 'max': # FixedGS_MAX
                            gs = pipeline.gs_max 
                        elif self.gs_type == 'min': # FixedGS_Min
                            gs = pipeline.gs_min
                    
                    L_concept = torch.tensor(0.); L_bg = torch.tensor(0.)
                    L_merge, grad, t_step = pipeline.sds_default(
                        pred_rgb = raster_img.to(self.weight_dtype),
                        im_size = model2res(sd_model_cfg.model_id),
                        prompts=gen_prompts,
                        negative_prompts=negative_prompts,
                        step = self.step, 
                        grad_scale=guidance_cfg.grad_scale,  
                        guidance_scale_ = gs
                    )         

               # Xing Loss for Self-Interaction Problem
                L_add = torch.tensor(0.)
                if self.style == "iconography" or self.x_cfg.xing_loss.use:
                    for r in renderers:
                        L_add += xing_loss_fn(r.get_point_parameters()) * self.x_cfg.xing_loss.weight
                
                loss = L_merge + L_concept + self.bg_lambda * L_bg + L_add

                # optimization
                for opt_ in optimizers:
                    opt_.zero_grad_()
                loss.backward()
                for opt_ in optimizers:
                    opt_.step_()

                # curve regularization
                for r in renderers:
                    r.clip_curve_shape()

                # re-init paths
                path_reinit.stop_step = 700
                if path_reinit.use and self.step % path_reinit.freq == 0 and self.step < path_reinit.stop_step and self.step != 0:

                    for i, r in enumerate(renderers):
                        extra_point_params, extra_color_params, extra_width_params = \
                            r.reinitialize_paths(f"P{batch_idx} - Step {self.step}",
                                                 path_reinit.opacity_threshold,
                                                 path_reinit.area_threshold)
                        optimizers[i].add_params(extra_point_params, extra_color_params, extra_width_params)

                # update lr
                if self.vpsd_optim.lr_schedule:
                    for opt_ in optimizers:
                        opt_.update_lr()

                # log pretrained model lr
                lr_str = ""
                for k, lr in optimizers[0].get_lr().items():
                    lr_str += f"{k}_lr: {lr:.4f}, "

                pbar.set_description(
                    lr_str +
                    f"t: {t_step.item():.2f}, "
                    f"L_total: {loss.item():.4f}, "
                    f"L_add: {L_add.item():.4e}, "
                    f"grad: {grad.item():.4e}, "
                )
                
                # self.args.save_step = 100
                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    bg_save_svg_path = self.save_svg_png(renderers[0], name = 'bg', step = self.step, i = batch_idx)
                    obj_save_svg_path = self.save_svg_png(renderers[1], 'obj', step = self.step, i = batch_idx)
                    merged_svg_path = self.sd_svg_logs_dir / f'{batch_idx}_iter{self.step}.svg'
                    merge_svg_files(
                        svg_path_1=bg_save_svg_path,
                        svg_path_2=obj_save_svg_path,
                        merge_type='simple',
                        output_svg_path=merged_svg_path.as_posix(),
                        out_size=(self.im_size, self.im_size)
                    )
                    merged_png_path = self.sd_png_logs_dir / f'{batch_idx}_iter{self.step}.png'
                    cairosvg.svg2png(url=merged_svg_path.as_posix(), write_to=merged_png_path.as_posix())
                    
                raster_img_prev = raster_img.clone()
                self.step += 1
                pbar.update(1)

        # df = pd.DataFrame(data)
        # file_path = self.result_path / "data.xlsx"
        # df.to_excel(file_path, index=False)

        # save final        
        bg_save_svg_path = self.save_svg_png(renderers[0], 'bg', step = self.step, i = batch_idx)
        obj_save_svg_path = self.save_svg_png(renderers[1], 'obj', step = self.step, i = batch_idx)
        merged_svg_path = self.result_path / f'{batch_idx}/Final.svg'
        merge_svg_files(
            svg_path_1=bg_save_svg_path,
            svg_path_2=obj_save_svg_path,
            merge_type='simple',
            output_svg_path=merged_svg_path.as_posix(),
            out_size=(self.im_size, self.im_size)
        )
        merged_png_path = self.result_path / f'{batch_idx}/Final.png'
        cairosvg.svg2png(url=merged_svg_path.as_posix(), write_to=merged_png_path.as_posix())

        del pipeline
        torch.cuda.empty_cache()

        # df = pd.DataFrame(debug_norm_list)
        # df.to_csv(self.result_path / "norm.csv")

    def save_svg_png(self, renderer, name, step, i):
        
        if name == 'bg': 
            dst_svg = self.sd_bg_svg_logs_dir
            dst_png = self.sd_bg_png_logs_dir
        elif name == 'obj':
            dst_svg = self.sd_obj_svg_logs_dir
            dst_png = self.sd_obj_png_logs_dir
        
        renderer.pretty_save_svg(dst_svg / f"{name}_svg_iter{step}_p{i}.svg")
        image = renderer.get_image()
        torchvision.utils.save_image(image, 
                    fp=dst_png / f'{name}_iter{step}.png')
        return dst_svg / f"{name}_svg_iter{step}_p{i}.svg"
    
    def vis_image(self, img, fpath): 
        img = img.squeeze(0)
        img_np = img.permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(self.result_path / fpath)
                    
    def get_combined_img(self, bg_tensor, obj_tensor):
        para_bg = torch.tensor([1., 1., 1.], device=self.device)

        img = bg_tensor + obj_tensor
        img = torch.clamp(img, min = 0, max = 1)
        img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img
    
    def alpha_blend(self, bg_tensor, obj_tensor):
        bg_rgb, bg_alpha = bg_tensor[:, :, :3], bg_tensor[:, :, 3:4]
        obj_rgb, obj_alpha = obj_tensor[:, :, :3], obj_tensor[:, :, 3:4]
        
        blended_rgb = obj_alpha * obj_rgb + (1 - obj_alpha) * bg_rgb
        blended_alpha = obj_alpha + (1 - obj_alpha) * bg_alpha

        blended_image = torch.cat([blended_rgb, blended_alpha], dim=-1)

        para_bg = torch.tensor([1., 1., 1.], device=self.device)
        blended_image = blended_image[:, :, 3:4] * blended_image[:, :, :3] + para_bg * (1 - blended_image[:, :, 3:4])

        # 텐서 형태 변환
        blended_image = blended_image.unsqueeze(0)  # HWC -> NHWC
        blended_image = blended_image.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return blended_image
    

    def load_renderer(self, path_svg=None):
        renderer = Painter(self.args.diffvg,
                           self.style,
                           self.x_cfg.num_segments,
                           self.x_cfg.segment_init,
                           self.x_cfg.radius,
                           self.im_size,
                           self.x_cfg.grid,
                           self.x_cfg.trainable_bg,
                           self.x_cfg.width,
                           path_svg=path_svg,
                           device=self.device)
        return renderer

    def apply_lpf_to_rgb(self, image, cutoff_frequency=30): # low pass filter function 
        
        b, g, r = cv2.split(image) # b,g,r
        
        def lpf_channel(channel, cutoff):
            f_transform = fftshift(fft2(channel))
            
            rows, cols = channel.shape
            crow, ccol = rows // 2 , cols // 2

            # LPF mask 
            mask = np.zeros((rows, cols), dtype=np.float32)
            mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
            
            # apply filter 
            filtered_transform = f_transform * mask
            filtered_channel = np.abs(ifft2(ifftshift(filtered_transform)))
            
            # scaling 
            filtered_channel = (filtered_channel - filtered_channel.min()) / (filtered_channel.max() - filtered_channel.min())
            filtered_channel = (filtered_channel * 255).astype(np.uint8)

            return filtered_channel

        b_filtered = lpf_channel(b, cutoff_frequency)
        g_filtered = lpf_channel(g, cutoff_frequency)
        r_filtered = lpf_channel(r, cutoff_frequency)
        
        filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))
        return filtered_image

    def apply_hpf_to_rgb(self, image, cutoff_frequency=30): # high path filter 
       
        b, g, r = cv2.split(image)
        
        def hpf_channel(channel, cutoff):
            f_transform = fftshift(fft2(channel))
            
            rows, cols = channel.shape
            crow, ccol = rows // 2 , cols // 2
            mask = np.ones((rows, cols), dtype=np.float32)
            mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0  # low frequency out 
            
            filtered_transform = f_transform * mask
            filtered_channel = np.abs(ifft2(ifftshift(filtered_transform)))
            return filtered_channel

        b_filtered = hpf_channel(b, cutoff_frequency)
        g_filtered = hpf_channel(g, cutoff_frequency)
        r_filtered = hpf_channel(r, cutoff_frequency)
        
        filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))
        return filtered_image


    def FIVE_stage(self,
                   text_prompt: str, 
                   batch_idx: int, 
                   select_sample_path:str = None):
        
        self.print(f"Text Prompt: {text_prompt}")
        self.print(f"Token Index: {self.args.token_ind}")

        # extract attention map 
        select_sample_path = self.result_path / f'{batch_idx}/pixel_sample.png'
        pipeline = DiffusionPipeline(self.x_cfg.sive_model_cfg, self.args.diffuser, self.device)
        attn_map_fpath = self.extract_ldm_attn(batch_idx, 
                                               self.x_cfg.sive_model_cfg,
                                               pipeline, 
                                               text_prompt, 
                                               select_sample_path, 
                                               self.sive_cfg.attn_cfg, 
                                               self.im_size, 
                                               self.args.token_ind)

        # load pixel image
        select_img = self.target_file_preprocess(select_sample_path)
        image = cv2.imread(select_sample_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # frequency analysis
        lpf_image = self.apply_lpf_to_rgb(image_rgb, cutoff_frequency=30)
        hpf_image = self.apply_hpf_to_rgb(image_rgb, cutoff_frequency=30)

        # save frequency analysis as image
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Image") 
        plt.imshow(image_rgb)
        plt.subplot(1, 3, 2)
        plt.title("LPF Applied Image")
        plt.imshow(lpf_image.astype(np.uint8)) 
        plt.subplot(1, 3, 3)
        plt.title("HPF Applied Image (Edges)") 
        plt.imshow(hpf_image.astype(np.uint8)) 
        plt.savefig(self.five_freq_dir / f'{batch_idx}_lpf_hpf_results.png', format="png")
    
        lpf_image_uint8 = (lpf_image * 255).astype(np.uint8) if lpf_image.max() <= 1 else lpf_image.astype(np.uint8)
        hpf_image_uint8 = (hpf_image * 255).astype(np.uint8) if hpf_image.max() <= 1 else hpf_image.astype(np.uint8)

        merged_lpf_image = Image.fromarray(lpf_image_uint8)
        merged_hpf_image = Image.fromarray(hpf_image_uint8)
        merged_lpf_image.save(self.five_freq_dir/ f"{batch_idx}_lpf_output_image.jpg")  
        merged_hpf_image.save(self.five_freq_dir/ f"{batch_idx}_hpf_output_image.jpg")  
        
        # extract edge from hpf
        r_hpf, g_hpf, b_hpf= merged_hpf_image.split()
        hpf_combined = (np.abs(r_hpf) + np.abs(g_hpf) + np.abs(b_hpf)) / 3

        # Thresholding 
        _, edge_mask = cv2.threshold(hpf_combined, 50, 1, cv2.THRESH_BINARY)  
        edge_mask_img = Image.fromarray(np.uint8(edge_mask * 255))  
        edge_mask_img.save(self.five_freq_dir/ f"{batch_idx}_edge_output_image.jpg")
        
        # optimization 
        low_path = int(512 * 0.6)
        high_path = 512 - low_path
        
        low_target_img = self.target_file_preprocess(self.five_freq_dir/ f"{batch_idx}_lpf_output_image.jpg")
        render_low_path = self.component_rendering(tag=f'{batch_idx}_low',
                                    prompt=text_prompt,
                                    num_paths = low_path, 
                                    target_img = low_target_img,
                                    mask = None, 
                                    attention_map= None,
                                    canvas_size=(self.im_size, self.im_size),
                                    render_cfg=self.sive_cfg.bg,
                                    optim_cfg=self.sive_optim,
                                    log_png_dir=self.five_logs_dir,
                                    log_svg_dir=self.five_logs_dir)

        edge_img = Image.open(self.five_freq_dir/ f"{batch_idx}_edge_output_image.jpg")
        edge_img = edge_img.resize((600,600))
        edge = np.array(edge_img).astype(int) # (600, 600)
        edge = (edge - edge.min()) / (edge.max() - edge.min())

        bool_edge = edge > 0.3
        mask = bool_edge.astype(int)  # [w, h]

        def get_hpf_edge_image(select_sample_path, edge_path, save_path):
            img = Image.open(select_sample_path).convert('RGBA')  
            mask = Image.open(edge_path).convert('L')  

            expanded_mask = mask.filter(ImageFilter.MaxFilter(size=13)) 

            new_img = Image.composite(img, Image.new('RGBA', img.size, (0, 0, 0, 0)), expanded_mask)

            background = Image.new("RGB", new_img.size, (255, 255, 255))  
            final_img = Image.alpha_composite(background.convert('RGBA'), new_img).convert('RGB') 
            final_img.save(save_path)
        
        get_hpf_edge_image(select_sample_path= select_sample_path, 
                           edge_path= self.five_freq_dir/ f"{batch_idx}_edge_output_image.jpg",
                           save_path= self.five_freq_dir/ f"{batch_idx}_hpf_edge_image.jpg")
        hpf_edge_image = self.target_file_preprocess(self.five_freq_dir/ f"{batch_idx}_hpf_edge_image.jpg")
        
        render_high_path = self.component_rendering(tag=f'{batch_idx}_high',
                                    prompt= text_prompt, 
                                    num_paths = high_path, 
                                    target_img = hpf_edge_image,
                                    mask = mask, 
                                    attention_map= None,
                                    canvas_size=(self.im_size, self.im_size),
                                    render_cfg=self.sive_cfg.fg,
                                    optim_cfg=self.sive_optim,
                                    log_png_dir=self.five_logs_dir,
                                    log_svg_dir=self.five_logs_dir)
        
        self.print(f"-> merge high and low ")
        merged_render_path = self.result_path / f'{batch_idx}/FIVE_render_final.svg'
        merge_svg_files(
            svg_path_1=render_low_path,
            svg_path_2=render_high_path,
            merge_type='simple',
            output_svg_path=merged_render_path.as_posix(),
            out_size=(self.im_size, self.im_size)
        )
        # svg-to-png, to tensor
        merged_png_path = self.result_path / f'{batch_idx}/FIVE_render_final.png'
        cairosvg.svg2png(url=merged_render_path.as_posix(), write_to=merged_png_path.as_posix())

        # foreground and background refinement
        # Note: you are not allowed to add further paths here
        if self.sive_cfg.tog.reinit:
            self.print("-> enable vector graphic refinement:")
            merged_render_path = self.refine_rendering(tag=f'{batch_idx}_refine',
                                                    prompt=text_prompt,
                                                    target_img=select_img,
                                                    canvas_size=(self.im_size, self.im_size),
                                                    render_cfg=self.sive_cfg.tog,
                                                    optim_cfg=self.sive_optim,
                                                    init_svg_path=merged_render_path)
            # svg-to-png, to tensor
            merged_png_path = self.result_path / f'{batch_idx}/FIVE_render_final.png'
            cairosvg.svg2png(url=merged_render_path.as_posix(), write_to=merged_png_path.as_posix())

        return merged_render_path, attn_map_fpath, select_sample_path

    def refine_rendering(self,
                         tag: str,
                         prompt: str,
                         target_img: torch.Tensor,
                         canvas_size: Tuple[int, int],
                         render_cfg: omegaconf.DictConfig,
                         optim_cfg: omegaconf.DictConfig,
                         init_svg_path: str):
        # init renderer
        content_renderer = CompPainter(self.style,
                                       target_img,
                                       path_svg=init_svg_path,
                                       canvas_size=canvas_size,
                                       device=self.device)
        # init graphic
        img = content_renderer.init_image()
        plot_img(img, self.five_png_logs_dir, fname=f"{tag}_before_refined")

        n_iter = render_cfg.num_iter

        # build painter optimizer
        optimizer = CompPainterOptimizer(content_renderer, self.style, n_iter, optim_cfg)
        # init optimizer
        optimizer.init_optimizers()

        print(f"=> n_point: {len(content_renderer.get_point_params())}, "
              f"n_width: {len(content_renderer.get_width_params())}, "
              f"n_color: {len(content_renderer.get_color_params())}")

        step = 0
        loss_weight_keep = 0
        with tqdm(initial=step, total=n_iter, disable=not self.accelerator.is_main_process) as pbar:
            for t in range(n_iter):
                raster_img = content_renderer.get_image(step=t).to(self.device)

                loss_recon = F.mse_loss(raster_img, target_img)

                # udf  
                loss_weight = content_renderer.calc_distance_weight(loss_weight_keep)
                    
                loss_weight = content_renderer.calc_distance_weight(loss_weight_keep)
                loss_udf = ((raster_img - target_img) ** 2)
                loss_udf = (loss_udf.sum(1) * loss_weight).mean()

                loss = loss_recon + loss_udf

                lr_str = ""
                for k, lr in optimizer.get_lr().items():
                    lr_str += f"{k}_lr: {lr:.4f}, "

                pbar.set_description(lr_str + f"L_refine: {loss.item():.4f}")

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                content_renderer.clip_curve_shape()

                if step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(target_img,
                                raster_img,
                                step,
                                prompt=prompt,
                                output_dir=self.five_png_logs_dir.as_posix(),
                                fname=f"{tag}_iter{step}")
                    content_renderer.save_svg(self.five_svg_logs_dir / f"{tag}_svg_iter{step}.svg")

                step += 1
                pbar.update(1)

        # update current svg
        content_renderer.save_svg(init_svg_path)
        # save
        img = content_renderer.get_image()
        plot_img(img, self.five_png_logs_dir, fname=f"{tag}_refined")

        return init_svg_path
   
    def component_rendering(self,
                            tag: str,
                            prompt: AnyPath,
                            num_paths: int, 
                            target_img: torch.Tensor,
                            mask: Union[np.ndarray, None],
                            attention_map: Union[np.ndarray, None],
                            canvas_size: Tuple[int, int],
                            render_cfg: omegaconf.DictConfig,
                            optim_cfg: omegaconf.DictConfig,
                            log_png_dir: pathlib.Path,
                            log_svg_dir: pathlib.Path):

        # set path_schedule
        each = 32
        step = num_paths // each 

        print(num_paths, each)

        if num_paths % each == 0 : 
            path_schedule = [each] * (step)
        else: 
            path_schedule = [each] * step + [num_paths % each]
        
        self.print(f'-> set path schedule {path_schedule}')
        
        if render_cfg.style == 'pixelart':
            path_schedule = [render_cfg.grid]
        self.print(f"path_schedule: {path_schedule}")

        # for convenience
        n_iter = render_cfg.num_iter
        style = render_cfg.style
        trainable_bg = render_cfg.optim_bg
        total_step = len(path_schedule) * n_iter

        # set renderer
        renderer = CompPainter(style,
                               target_img,
                               canvas_size,
                               render_cfg.num_segments,
                               render_cfg.segment_init,
                               render_cfg.radius,
                               render_cfg.grid,
                               render_cfg.width,
                               device=self.device,
                               attn_init=render_cfg.use_attn_init and attention_map is not None,
                               attention_map=attention_map,
                               attn_prob_tau=render_cfg.softmax_tau)

        if mask is not None:
            select_inds = renderer.init_points_mask(num_paths= sum(path_schedule), mask = mask)
            plot_attn2(target_img, select_inds,
                    (self.five_freq_dir / f"{tag}_map.jpg").as_posix())
            renderer.attn_init = True
        else: 
            renderer.component_wise_path_init(pred=target_img, init_type=render_cfg.coord_init)

        optimizer_list = [
            CompPainterOptimizer(renderer, style, n_iter, optim_cfg, trainable_bg)
            for _ in range(len(path_schedule))
        ]

        pathn_record = []
        render_cfg.use_distance_weighted_loss = True ## udf 
        loss_weight_keep = 0
        step = 0
        loss_weight = 1
        with tqdm(initial=step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            for path_idx, pathn in enumerate(path_schedule):
                # record path
                pathn_record.append(pathn)
                # init graphic
                img = renderer.init_image(num_paths=pathn)

                # rebuild optimizer
                optimizer_list[path_idx].init_optimizers(pid_delta=int(path_idx * pathn))

                pbar.write(f"=> adding {pathn} paths, n_path: {sum(pathn_record)}, "
                           f"n_point: {len(renderer.get_point_params())}, "
                           f"n_width: {len(renderer.get_width_params())}, "
                           f"n_color: {len(renderer.get_color_params())}")

                for t in range(n_iter):
                    raster_img = renderer.get_image(step=t).to(self.device)
                    
                    # reconstruction loss
                    loss_recon = F.mse_loss(raster_img, target_img)

                    # Xing Loss for Self-Interaction Problem
                    loss_xing = torch.tensor(0.)
                    if style == "iconography":
                        loss_xing = xing_loss_fn(renderer.get_point_params()) * render_cfg.xing_loss_weight

                    # udf 버전 
                    if render_cfg.use_distance_weighted_loss and style == "iconography":
                        loss_weight = renderer.calc_distance_weight(loss_weight_keep)
                        
                    loss_weight = renderer.calc_distance_weight(loss_weight_keep)
                    loss_udf = ((raster_img - target_img) ** 2)
                    loss_udf = (loss_udf.sum(1) * loss_weight).mean()

                    # total loss
                    loss = loss_xing + loss_udf

                    lr_str = ""
                    for k, lr in optimizer_list[path_idx].get_lr().items():
                        lr_str += f"{k}_lr: {lr:.4f}, "

                    pbar.set_description(
                        lr_str +
                        f"L_total: {loss.item():.4f}, "
                        f"L_recon: {loss_recon.item():.4f}, "
                        f"L_udf: {loss_udf.item():.4f}, "
                        f"L_xing: {loss_xing.item():.4e}"
                    )

                    # optimization
                    for i in range(path_idx + 1):
                        optimizer_list[i].zero_grad_()

                    loss.backward()

                    for i in range(path_idx + 1):
                        optimizer_list[i].step_()

                    renderer.clip_curve_shape()

                    if render_cfg.lr_schedule:
                        for i in range(path_idx + 1):
                            optimizer_list[i].update_lr()

                    if step % 50 == 0 and self.accelerator.is_main_process:
                        plot_couple(target_img,
                                    raster_img,
                                    step,
                                    prompt=prompt,
                                    output_dir=log_png_dir.as_posix(),
                                    fname=f"{tag}_iter{step}")
                        renderer.save_svg(log_svg_dir / f"{tag}_svg_iter{step}.svg")

                    step += 1
                    pbar.update(1)

                if render_cfg.use_distance_weighted_loss and style == "iconography":
                    loss_weight_keep = loss_weight.detach().cpu().numpy() * 1
                # calc center
                renderer.component_wise_path_init(raster_img)

        # end LIVE
        plot_couple(target_img,
            raster_img,
            step,
            prompt=prompt,
            output_dir=log_png_dir.as_posix(),
            fname=f"{tag}_iter_final")
                                
        final_svg_fpth = self.five_final_dir / f"{tag}_final_render.svg"
        final_png_fpth = self.five_final_dir / f"{tag}_final_render.png"
        renderer.save_svg(final_svg_fpth)
        cairosvg.svg2png(url=final_svg_fpth.as_posix(), write_to=final_png_fpth.as_posix())

        return final_svg_fpth

    def get_path_schedule(self,
                          path_schedule: str,
                          schedule_each: Union[int, List],
                          num_paths: int = None):
        if path_schedule == 'repeat':
            assert num_paths is not None
            return int(num_paths / schedule_each) * [schedule_each]
        elif path_schedule == 'list':
            assert isinstance(schedule_each, list) or isinstance(schedule_each, omegaconf.ListConfig)
            return schedule_each
        else:
            raise NotImplementedError

    def extract_ldm_attn(self,
                         iter: int,
                         model_cfg: omegaconf.DictConfig,
                         pipeline: DiffusionPipeline,
                         prompts: str,
                         gen_sample_path: AnyPath,
                         attn_init_cfg: omegaconf.DictConfig,
                         image_size: int,
                         token_ind: int,
                         attn_init: bool = True):
        if token_ind <= 0:
            raise ValueError("The 'token_ind' should be greater than 0")

        # init controller
        controller = AttentionStore() if attn_init else EmptyControl()

        # forward once and record attention map
        height = width = model2res(model_cfg.model_id)
        outputs = pipeline.sample(prompt=[prompts],
                                  height=height,
                                  width=width,
                                  num_inference_steps=model_cfg.num_inference_steps,
                                  controller=controller,
                                  guidance_scale=model_cfg.guidance_scale,
                                  negative_prompt=self.args.neg_prompt,
                                  generator=self.g_device)
        outputs_np = [np.array(img) for img in outputs.images]
        view_images(outputs_np, save_image=True, fp=gen_sample_path)
        self.print(f"select_sample shape: {outputs_np[0].shape}")

        """ldm cross-attention map"""
        cross_attention_maps, tokens = \
            pipeline.get_cross_attention([prompts],
                                         controller,
                                         res=attn_init_cfg.cross_attn_res,
                                         from_where=("up", "down"),
                                         save_path=self.five_attn_dir / f"cross-attn-{iter}.png")

        self.print(f"the length of tokens is {len(tokens)}, select {token_ind}-th token")
        # [res, res, seq_len]
        self.print(f"origin cross_attn_map shape: {cross_attention_maps.shape}")
        # [res, res]
        cross_attn_map = cross_attention_maps[:, :, token_ind]
        self.print(f"select cross_attn_map shape: {cross_attn_map.shape}")
        cross_attn_map = 255 * cross_attn_map / cross_attn_map.max()
        # [res, res, 3]
        cross_attn_map = cross_attn_map.unsqueeze(-1).expand(*cross_attn_map.shape, 3)
        # [3, res, res]
        cross_attn_map = cross_attn_map.permute(2, 0, 1).unsqueeze(0)
        # [3, clip_size, clip_size]
        cross_attn_map = F.interpolate(cross_attn_map, size=image_size, mode='bicubic')
        cross_attn_map = torch.clamp(cross_attn_map, min=0, max=255)
        # rgb to gray
        cross_attn_map = rgb2gray(cross_attn_map.squeeze(0).permute(1, 2, 0)).astype(np.float32)
        # torch to numpy
        if cross_attn_map.shape[-1] != image_size and cross_attn_map.shape[-2] != image_size:
            cross_attn_map = cross_attn_map.reshape(image_size, image_size)
        # to [0, 1]
        attn_map = (cross_attn_map - cross_attn_map.min()) / (cross_attn_map.max() - cross_attn_map.min())

        # visual fusion-attention
        attn_map_vis = np.copy(attn_map)
        attn_map_vis = attn_map_vis * 255
        attn_map_vis = np.repeat(np.expand_dims(attn_map_vis, axis=2), 3, axis=2).astype(np.uint8)
        attn_map_fpath = self.five_attn_dir / f'fusion-attn-{iter}.png'
        view_images(attn_map_vis, save_image=True, fp=attn_map_fpath)

        self.print(f"-> fusion attn_map: {attn_map.shape}")

        return attn_map_fpath

    def target_file_preprocess(self, tar_path: AnyPath):
        process_comp = transforms.Compose([
            transforms.Resize(size=(self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])

        tar_pil = Image.open(tar_path).convert("RGB")  # open file
        target_img = process_comp(tar_pil)  # preprocess
        target_img = target_img.to(self.device)
        return target_img

    def target_file_preprocess_(self, tar_path: AnyPath):
        process_comp = transforms.Compose([
            transforms.Resize(size=(self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])

        tar_pil = Image.open(tar_path).convert("RGBA")  # open file
        target_img = process_comp(tar_pil)  # preprocess
        target_img = target_img.to(self.device)
        return target_img
