import os
import platform

from PIL import Image, PngImagePlugin
from modules import devices, shared

import gc

import numpy as np
from stripper_image_editing import scale_image, combine_images, auto_resize_to_pil as resize_to_pil, create_borderd_image

import random
from tqdm import tqdm

from stripper_videodata import VideoData

import cv2
import torch
from diffusers import (DDIMScheduler,
                       KDPM2AncestralDiscreteScheduler,
                       StableDiffusionInpaintPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline, StableDiffusionUpscalePipeline)

#Controlnet
from controlnet_aux import OpenposeDetector
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

from check_versions import check_versions

from datetime import datetime


output_parent_folder = "output"
output_folder = f"{output_parent_folder}/stripper"

if not os.path.exists(output_parent_folder):
    os.mkdir(output_parent_folder)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

#Logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False

def generate_controlnet_inapint_images(image, mask, pose, prompt:str, n_prompt:str, sampling_steps:int, cfg_scale:float, seed:int=-1, iteration_count:int=1, inpaint_model_id:str="Uminosachi/realisticVisionV51_v51VAE-inpainting", inpaint_resolution:int=1000):
    save_folder_path = get_save_folder()

    original_height, original_width = image.shape[:2]

    input_image = scale_image(image, inpaint_resolution)
    mask_image = scale_image(mask, inpaint_resolution)
    pose_image = scale_image(pose, inpaint_resolution)

    input_image, mask_image, pose_image = resize_to_pil([input_image, mask_image, pose_image])
    width, height = input_image.size


    input_image.save(f"{save_folder_path}/scaled input image.jpg")
    mask_image.save(f"{save_folder_path}/scaled mask image.jpg")
    pose_image.save(f"{save_folder_path}/scaled pose image.jpg")

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(inpaint_model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False, local_files_only=False)
    pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    if seed == -1:
        seed = random.randint(0, 2147483647)

    output_images = []
    for i in range(iteration_count):
        image_seed = seed + i
        gc.collect()

        generator = torch.Generator(device="cuda").manual_seed(image_seed)

        # generate image
        output_image = pipe(
            prompt = prompt,
            negative_prompt = n_prompt,
            num_inference_steps=sampling_steps,
            generator=generator,
            image=input_image,
            width=width,
            height=height,
            mask_image=mask_image,
            control_image=pose_image
        ).images[0]

        output_image.save(f"{save_folder_path}/{i + 1}:generated image.jpg")

        composite_image = combine_images(image, output_image, mask)

        composite_image.save(f"{save_folder_path}/{i + 1}:composite image.jpg")

        output_images.append(composite_image)

        yield output_images

def generate_controlnet_outpaint_images(image, outpaint_up:int, outpaint_down:int, outpaint_left:int, outpaint_right:int, prompt:str, n_prompt:str, sampling_steps:int, cfg_scale:float, seed:int=-1, iteration_count:int=1, inpaint_model_id:str="Uminosachi/realisticVisionV51_v51VAE-inpainting", inpaint_resolution:int=1000):
    save_folder_path = get_save_folder()

    bordered_image, mask = create_borderd_image(image, outpaint_up, outpaint_down, outpaint_left, outpaint_right)

    input_image = scale_image(bordered_image, inpaint_resolution)
    mask_image = scale_image(mask, inpaint_resolution)
    pose_image = np.array(openpose(np.array(input_image, np.uint8), hand_and_face=True))

    input_image, mask_image, pose_image = resize_to_pil([input_image, mask_image, pose_image])
    width, height = input_image.size

    input_image.save(f"{save_folder_path}/scaled input image.jpg")
    mask_image.save(f"{save_folder_path}/scaled mask image.jpg")
    pose_image.save(f"{save_folder_path}/scaled pose image.jpg")

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(inpaint_model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False, local_files_only=False)
    pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    if seed == -1:
        seed = random.randint(0, 2147483647)

    output_images = []
    for i in range(iteration_count):
        image_seed = seed + i
        gc.collect()

        generator = torch.Generator(device="cuda").manual_seed(image_seed)

        # generate image
        output_image = pipe(
            prompt = prompt,
            negative_prompt = n_prompt,
            num_inference_steps=sampling_steps,
            generator=generator,
            image=input_image,
            width=width,
            height=height,
            mask_image=mask_image,
            control_image=pose_image
        ).images[0]


        composite_image = combine_images(bordered_image, output_image, mask)

        output_image.save(f"{save_folder_path}/{i + 1}:generated image.jpg")
        composite_image.save(f"{save_folder_path}/{i + 1}:composite image.jpg")

        output_images.append(composite_image)

        yield output_images

def generate_controlnet_vidoe_inpaint(video_path:str, mask_video_path:str, prompt:str, n_prompt:str, ddim_steps:int, cfg_scale:float):
    video = VideoData(video_path)
    mask = VideoData(mask_video_path)

def generate_inapint_images(image, mask, prompt:str, n_prompt:str, sampling_steps:int, cfg_scale:float, iteration_count: int = 1):
    sampler_name:str = "DPM2 a Karras"
    inpaint_model_id:str = "Uminosachi/realisticVisionV51_v51VAE-inpainting"

    input_image = scale_image(image)
    mask_image = scale_image(mask)

    init_image, mask_image = resize_to_pil([input_image, mask_image])
    width, height = init_image.size

    save_folder_path = get_save_folder()

    pipe, torch_generator = create_inpaint_pipline(inpaint_model_id, sampler_name)

    result_images = []
    iteration_count = iteration_count if iteration_count is not None else 1
    for i in range(iteration_count):
        seed = random.randint(0, 2147483647)
        gc.collect()

        generator = torch_generator.manual_seed(seed)

        pipe_args_dict = {
            "prompt": prompt,
            "image": init_image,
            "width": width,
            "height": height,
            "mask_image": mask_image,
            "num_inference_steps": sampling_steps,
            "guidance_scale": cfg_scale,
            "negative_prompt": n_prompt,
            "generator": generator,
        }

        output_image = pipe(**pipe_args_dict).images[0]

        image_save_path = f"{save_folder_path}/{i+1}:{prompt[:50] if len(prompt) > 50 else prompt} ({seed} ({seed}).jpg"

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Description", f"Prompt: {prompt} \n" +
                                         f"Negative Prompt: {n_prompt} \n" +
                                         f"Seed: {seed} \n" +
                                         f"Sampler: {sampler_name} \n" +
                                         f"Sampling Steps: {sampling_steps} \n"
                                         f"CFG Scale: {cfg_scale} \n" +
                                         f"Model: {inpaint_model_id} \n")

        output_image = combine_images(image, output_image, mask)
        output_image.save(image_save_path)

        result_images.append(output_image)
        
        yield result_images

def generate_outpaint_images(image, outpaint_up:int, outpaint_down:int, outpaint_left:int, outpaint_right:int, prompt:str, n_prompt:str, sampling_steps:int, cfg_scale:float, iteration_count: int = 1):
    sampler_name:str = "DPM2 a Karras"
    inpaint_model_id:str = "Uminosachi/realisticVisionV51_v51VAE-inpainting"

    image = Image.fromarray(image)
    
    bordered_image, mask = create_borderd_image(image, outpaint_up, outpaint_down, outpaint_left, outpaint_right)

    input_image = scale_image(bordered_image)
    mask_image = scale_image(mask)

    result_images = []

    init_image, mask_image = resize_to_pil([input_image, mask_image])
    width, height = init_image.size

    save_folder_path = get_save_folder()

    pipe, torch_generator = create_inpaint_pipline(inpaint_model_id, sampler_name)

    iteration_count = iteration_count if iteration_count is not None else 1
    for i in range(iteration_count):
        seed = random.randint(0, 2147483647)
        gc.collect()

        generator = torch_generator.manual_seed(seed)

        pipe_args_dict = {
            "prompt": prompt,
            "image": init_image,
            "width": width,
            "height": height,
            "mask_image": mask_image,
            "num_inference_steps": sampling_steps,
            "guidance_scale": cfg_scale,
            "negative_prompt": n_prompt,
            "generator": generator,
        }

        output_image = pipe(**pipe_args_dict).images[0]

        image_save_path = f"{save_folder_path}/{i+1}:{prompt[:50] if len(prompt) > 50 else prompt} ({seed} ({seed}).jpg"

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Description", f"Prompt: {prompt} \n" +
                                         f"Negative Prompt: {n_prompt} \n" +
                                         f"Seed: {seed} \n" +
                                         f"Sampler: {sampler_name} \n" +
                                         f"Sampling Steps: {sampling_steps} \n"
                                         f"CFG Scale: {cfg_scale} \n" +
                                         f"Model: {inpaint_model_id} \n")

        output_image = combine_images(bordered_image, output_image, mask)
        output_image.save(image_save_path)

        result_images.append(output_image)
       
        yield result_images

def run_video_inpaint(video_path:str, mask_video_path:str, prompt:str, n_prompt:str, ddim_steps:int, cfg_scale):
    video = VideoData(video_path)
    mask = VideoData(mask_video_path)
    inpaint_model_id:str = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
    seed = random.randint(0, 2147483647)

    output_path = f"{output_folder}/inpaint_video.mp4"

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video_writer = cv2.VideoWriter(output_path, fourcc, video.fps, (video.width, video.height))

    pbar = tqdm(total=video.frame_count, unit="frame", unit_scale=True)
    for i in range(video.frame_count):
        video_frame = cv2.cvtColor(video.frames[i], cv2.COLOR_BGR2RGB)
        mask_frame = None

        if i < mask.frame_count:
            mask_frame = cv2.cvtColor(mask.frames[i], cv2.COLOR_BGR2RGB)

        result_frame = None

        if mask_frame is not None:
            #Mask is not blank
            if not np.all(mask_frame == 0):
                result_frame = generate_inapint_images(video_frame, mask_frame, prompt, n_prompt, ddim_steps, cfg_scale, seed, inpaint_model_id)
        
        result_frame = np.array(result_frame)

        # Ensure result_frame is a numpy array and has the correct dtype and shape
        if result_frame.dtype == np.uint8 and result_frame.shape == (video.height, video.width, 3):
            video_writer.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
        else:
            print(f"Skipping frame {i} due to incorrect format or type")

        pbar.update(1)

    video_writer.release()
    return [output_path]

def get_webui_setting(key, default):
    value = shared.opts.data.get(key, default)

    if not isinstance(value, type(default)):
        value = default

    return value

def get_save_folder() -> str:
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    save_folder_path = f"{output_folder}/{formatted_time}"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    return save_folder_path

def create_inpaint_pipline(inpaint_model_id:str, sampler_name:str):
    torch_dtype = get_torch_dtype()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(inpaint_model_id, torch_dtype=torch_dtype)
    pipe.safety_checker = None

    pipe = set_sampler(pipe, sampler_name)

    if platform.system() == "Darwin":
        pipe = pipe.to("mps" if check_versions.torch_mps_is_available else "cpu")
        pipe.enable_attention_slicing()
        torch_generator = torch.Generator(devices.cpu)
    else:
        if check_versions.diffusers_enable_cpu_offload and devices.device != devices.cpu:
            logger.info("Enable model cpu offload")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(devices.device)
        if shared.xformers_available:
            logger.info("Enable xformers memory efficient attention")
            pipe.enable_xformers_memory_efficient_attention()
        else:
            logger.info("Enable attention slicing")
            pipe.enable_attention_slicing()
        if "privateuseone" in str(getattr(devices.device, "type", "")):
            torch_generator = torch.Generator(devices.cpu)
        else:
            torch_generator = torch.Generator(devices.device)
    return pipe, torch_generator

def set_sampler(pipeline: StableDiffusionInpaintPipeline, sampler_name:str):
    if sampler_name == "DPM2 a Karras":
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        logger.info("Sampler fallback to DDIM")
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    return pipeline
def get_torch_dtype():
    if platform.system() == "Darwin" or devices.device == devices.cpu or check_versions.torch_on_amd_rocm:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    return torch_dtype
