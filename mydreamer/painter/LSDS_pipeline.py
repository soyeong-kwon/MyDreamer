# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import re
from typing import Callable, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision import transforms
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

import math

class LSDSPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        try:
            num_channels_latents = self.unet.config.in_channels
        except Exception or Warning:
            num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. inherit TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        # do_denormalize = [True] * image.shape[0]
        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # 9. Run safety checker
        has_nsfw_concept = None
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def encode_(self, images):
        images = (2 * images - 1).clamp(-1.0, 1.0)  # images: [B, 3, H, W]

        # encode images
        latents = self.vae.encode(images).latent_dist.sample()
        latents = self.vae.config.scaling_factor * latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def x_augment(self, x: torch.Tensor, img_size: int = 512):
        augment_compose = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.5, p=0.7),
            transforms.RandomCrop(size=(img_size, img_size), pad_if_needed=True, padding_mode='reflect')
        ])
        return augment_compose(x)

    def schedule_timestep(self, step):
        min_step = int(self.num_train_timesteps * self.t_range[0])
        max_step = int(self.num_train_timesteps * self.t_range[1])
        if self.t_schedule == 'randint':
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        elif re.match(r"max_([\d.]+)_(\d+)", self.t_schedule):
            # Anneal time schedule
            # e.g: t_schedule == 'max_0.5_200'
            # [0.02, 0.98] -> [0.02, 0.5] after 200 steps
            tag, t_val, step_upd = str(self.t_schedule).split('_')
            t_val, step_upd = float(t_val), int(step_upd)
            if step >= step_upd:
                max_step = int(self.num_train_timesteps * t_val)
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        elif re.match(r"min_([\d.]+)_(\d+)", self.t_schedule):
            # Anneal time schedule
            # e.g: t_schedule == 'min_0.5_200'
            # [0.02, 0.98] -> [0.5, 0.98] after 200 steps
            tag, t_val, step_upd = str(self.t_schedule).split('_')
            t_val, step_upd = float(t_val), int(step_upd)
            if step >= step_upd:
                min_step = int(self.num_train_timesteps * t_val)
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        else:
            raise NotImplementedError(f"{self.t_schedule} is not support.")
        return t

    def get_text_embeddings(self, 
                            prompt, 
                            negative_prompt, 
                            batch_size = 2, 
                            guidance_scale = 100):
        
        ######## text condition 
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if negative_prompt is None:
                negative_prompt = [""] * batch_size

            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(self.device))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            
        return text_embeddings

    def predict_noise(self, latents, noise, time_step, embeddings, do_classifier_free_guidance = True, guidance_scale = 100):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # predict the noise residual with unet, stop gradient
        with torch.no_grad():
            # add noise
            latents_noisy = self.scheduler.add_noise(latents, noise, time_step)
            if guidance_scale > 1: 
                latent_model_input = torch.cat([latents_noisy] * 2).to(self.device)
            else: 
                latent_model_input = latents_noisy.to(self.device)
            noise_pred = self.unet(latent_model_input, time_step, encoder_hidden_states=embeddings).sample      
                    
            # perform guidance (high scale from paper!)
            if do_classifier_free_guidance:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                
        return noise_pred 

    def encode2latent(self, images):
        images = (2 * images - 1).clamp(-1.0, 1.0)  # images: [B, 3, H, W]
        # encode imagesf
        latents = self.vae.encode(images).latent_dist.sample()
        latents = self.vae.config.scaling_factor * latents
        return latents
    
    # def get_interactive_value(self, max, min, step, total_step):
    #     value = max - (max - min) * math.sqrt(step / total_step)
    #     return value

    def get_interactive_value(self, max_value, min_value, step, total_step, denoise_step, total_denoise_step):
        # Calculate the base value based on the range and step progress
        base_value = max_value - (max_value - min_value) * math.sqrt(step / total_step)

        try:
            # Apply a scaling factor based on the denoise step
            denoise_scale = 1 - (denoise_step / total_denoise_step)
        except: 
            denoise_scale = 1

        # Adjust the value with the denoise scale
        value = base_value * denoise_scale

        return value

    
    def sds_poc(self,
                pred_rgb: torch.Tensor,
                step: int,
                im_size: int,
                prompts: Union[List, str],
                ref_prompts: Union[List, str] = None,
                latents_o: torch.tensor = None, 
                negative_prompts: Union[List, str] = None,
                guidance_scale: float = 100,
                grad_scale: float = 1, 
                loss_type: int = 0, 
                alpha_range: str = '00',
                gs_range: str = '00',
                start_latent_interpolation: int = 300, 
                total_step: int = 1000): 

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        pred_rgb = self.x_augment(pred_rgb, im_size)  # input augmentation
        batch_size = pred_rgb.shape[0]
        
        # encode image into latents with vae, requires grad!
        pred_rgb_ = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latent_list = [self.encode2latent(pred_rgb_[i].unsqueeze(0)) for i in range(batch_size)]
        latents = torch.cat(latent_list, dim=0)
        latents = latents.to(self.device)

        noise = torch.randn_like(latents)
        
        t = self.schedule_timestep(step)
        text_embeddings = self.get_text_embeddings(prompt = prompts, 
                                                   negative_prompt= negative_prompts, 
                                                   batch_size = batch_size, 
                                                   guidance_scale= guidance_scale)

        if loss_type == 0: 
            noise_pred = self.predict_noise(latents= latents, 
                                noise = noise,
                                time_step= t, 
                                embeddings= text_embeddings, 
                                guidance_scale= 100) # vectorfusion은 guidance scale이 100 
            noise_final = noise_pred.clone()
        elif loss_type == 1: 
            
            if alpha_range != '00':
                alpha_max, alpha_min = list(map(float, alpha_range.split('_')))
                alpha_value = self.get_interactive_value(alpha_max, alpha_min, step = step, total_step= total_step, denoise_step= 0, total_denoise_step= 0)
            
            if gs_range != '00': 
                gs_max, gs_min = list(map(float, gs_range.split('_')))
                gs_value = self.get_interactive_value(gs_max, gs_min, step = step, total_step= total_step, denoise_step= 0, total_denoise_step= 0)
            
            # print(alpha_value, gs_value)
            
            noise_pred = self.predict_noise(latents= latents, 
                                            noise = noise,
                                            time_step= t, 
                                            embeddings= text_embeddings, 
                                            guidance_scale= gs_value)
            
            noise_pred_o = self.predict_noise(latents= latents_o, 
                                              noise = noise, 
                                              time_step= t, 
                                              embeddings= text_embeddings, 
                                              guidance_scale= gs_value)

            if step > start_latent_interpolation: 
                noise_final = (1- alpha_value) * noise_pred + alpha_value * noise_pred_o
            else: 
                noise_final = noise_pred.clone()
                
            
        w = (1 - alphas[t])
        grad = grad_scale * w * (noise_final - noise)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
 
        return loss, grad.mean(), t


    def score_distillation_sampling(self,
                                    pred_rgb: torch.Tensor,
                                    start_code: torch.Tensor,
                                    step: int,
                                    im_size: int,
                                    prompt: Union[List, str],
                                    negative_prompt: Union[List, str] = None,
                                    guidance_scale: float = 100,
                                    as_latent: bool = False,
                                    grad_scale: float = 1,
                                    t_range: Union[List[float], Tuple[float]] = (0.05, 0.95), 
                                    is_guided: bool = False):

        alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        # input augmentation
        pred_rgb_a = self.x_augment(pred_rgb, im_size)

        # the input is intercepted to im_size x im_size and then fed to the vae
        if as_latent:
            latents = F.interpolate(pred_rgb_a, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # encode image into latents with vae, requires grad!
            latents = self.encode_(pred_rgb_a)

        #  Encode input prompt
        num_images_per_prompt = 1  # the number of images to generate per prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings = self._encode_prompt(
            prompt, self.device, num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )

        # timestep ~ U(0.05, 0.95) to avoid very high/low noise level
        # t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        t = self.schedule_timestep(step)
        
        # predict the noise residual with unet, stop gradient
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents_noisy
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - alphas[t])
        grad = grad_scale * w * (noise_pred - noise)
        
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
 
        return loss, grad.mean(), t

    def score_distillation_sampling_0907(self,
                                    pred_rgb: torch.Tensor,
                                    start_code: torch.Tensor, 
                                    step: int,
                                    im_size: int,
                                    prompt: Union[List, str],
                                    ref_prompt: Union[List, str],
                                    negative_prompt: Union[List, str] = None,
                                    guidance_scale: float = 100,
                                    as_latent: bool = False,
                                    grad_scale: float = 1,
                                    is_sds: bool = False):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        pred_rgb_a = self.x_augment(pred_rgb, im_size)  # input augmentation

        # the input is intercepted to im_size x im_size and then fed to the vae
        if as_latent:
            latents = F.interpolate(pred_rgb_a, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # encode image into latents with vae, requires grad!
            latents = self.encode_(pred_rgb_a)

        def get_text_embeddings(prompt, negative_prompt, batch_size = 1, guidance_scale = guidance_scale):
            ######## text condition 
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )

            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            
            # unconditional embedding for classifier free guidance
            if guidance_scale > 1.:
                max_length = text_input.input_ids.shape[-1]
                if negative_prompt is None:
                    negative_prompt = [""] * batch_size

                # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
                unconditional_input = self.tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
                unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(self.device))[0]
                text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
                
            return text_embeddings
        
        def predict_noise(latents, time_step, embeddings, do_classifier_free_guidance = True, guidance_scale = guidance_scale):
            # predict the noise residual with unet, stop gradient
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)

                if guidance_scale > 1: 
                    latent_model_input = torch.cat([latents_noisy] * 2).to(self.device)
                else: 
                    latent_model_input = latents_noisy.to(self.device)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=embeddings).sample      
                     
                # perform guidance (high scale from paper!)
                if do_classifier_free_guidance:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            
            return noise_pred 
        
        t = self.schedule_timestep(step)
        
        text_embeddings = get_text_embeddings(prompt = prompt, negative_prompt= negative_prompt)
        noise_pred = predict_noise(latents= latents, time_step= t, embeddings= text_embeddings)
        noise_pred_pre = predict_noise(latents= latents, time_step = t - 20, embeddings= text_embeddings)
        
        # inv 
        text_embeddings_inv = get_text_embeddings(prompt = ref_prompt, negative_prompt= negative_prompt)
        noise_pred_inv = predict_noise(latents=start_code, time_step= t, embeddings= text_embeddings_inv)
        noise_pred_pre_inv = predict_noise(latents=start_code, time_step= t - 20, embeddings= text_embeddings_inv)
        
        
        w = (1 - alphas[t])
        grad = grad_scale * w * (noise_pred_pre - noise_pred)
        grad_inv = grad_scale * w * (noise_pred_pre_inv - noise_pred_inv)
        grad = grad + grad_inv
        
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
 
        return loss, grad.mean(), t

    
    
    def guided_score_distillation_sampling(self,
                                    pred_rgb: torch.Tensor,
                                    latent_ni: torch.Tensor, 
                                    step: int,
                                    im_size: int,
                                    prompt: Union[List, str],
                                    negative_prompt: Union[List, str] = None,
                                    uncond_embedding: torch.Tensor = None, 
                                    guidance_scale: float = 100,
                                    as_latent: bool = False,
                                    grad_scale: float = 1,
                                    t_range: Union[List[float], Tuple[float]] = (0.05, 0.95)):

        alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        # input augmentation
        pred_rgb_a = self.x_augment(pred_rgb, im_size)

        # the input is intercepted to im_size x im_size and then fed to the vae
        if as_latent:
            latents = F.interpolate(pred_rgb_a, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # encode image into latents with vae, requires grad!
            latents = self.encode_(pred_rgb_a)

        #  Encode input prompt
        num_images_per_prompt = 1  # the number of images to generate per prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings = self._encode_prompt(
            prompt, self.device, num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
    
        # timestep ~ U(0.05, 0.95) to avoid very high/low noise level
        # t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        t = self.schedule_timestep(step)

        # predict the noise residual with unet, stop gradient
        with torch.no_grad():
            ## add noise
            # original 
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents_noisy
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # w(t), sigma_t^2
        # epsilon_ni = (latents_noisy - alphas[t] * latent_ni) / (1 - alphas[t])
        w = (1 - alphas[t])
        # grad = grad_scale * w * (noise_pred - (latents_noisy + latent_ni)) 
        grad = grad_scale * w * (noise_pred - latent_ni)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
 
        return loss, grad.mean(), t
    

class SpecifyGradient(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None