import os
import platform

if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import gradio as gr
import numpy as np
import gc

import random
import cv2
from modules import script_callbacks
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import torch

#Stripper Libraries
from stripper_celebs import *
from stripper_prompt_generator import BOOB_SIZES, BODY_TYPES, ASS_SIZES, ABS_OPTIONS, generate_stripper_prompts
from stripper_find_webui_extensions import *
from stripper_image_generation import generate_outpaint_images, generate_inapint_images, generate_controlnet_inapint_images, generate_controlnet_outpaint_images, create_inpaint_pipline
from stripper_image_editing import scale_image, auto_resize_to_pil, combine_images
from controlnet_aux import OpenposeDetector

#ControlNet
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

pose_image = gr.Image(label="Pose", elem_id="stripper-pose-image", height=512)

celeb_name_dropdown = gr.Dropdown(label="Celeb Presets", choices=celeb_names, value=celeb_names[0])

inpaint_model_ids:list = ["Uminosachi/realisticVisionV51_v51VAE-inpainting", "Lykon/absolute-reality-1.6525-inpainting"]
inpaint_model_dropdown = gr.Dropdown(label="Inpaint Model", choices=inpaint_model_ids, value=inpaint_model_ids[0])

output_parent_folder = "output"
output_folder = f"{output_parent_folder}/stripper"

if not os.path.exists(output_parent_folder):
    os.mkdir(output_parent_folder)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


media_type_dropdown = gr.Dropdown(label="Media Type", choices=["Image", "Video"], value="Image")
generation_type_dropdown = gr.Dropdown(label="Generation Type", choices=["Inpaint", "Outpaint"], value="Inpaint")

input_image = gr.Image(label="Input Image", elem_id="stripper-input-image", elem_classes=["stripper-image"], source="upload", height=1000, interactive=True, visible=True)
input_video = gr.Video(label="Input Video", elem_id="stripper-input-video", elem_classes=["stripper-video"], source="upload", height=1000, interactive=True, visible=False)


#Inpaint
inpaint_resolution_slider = gr.Slider(label="Generation Resolution", elem_id="stripper-generation-resolution", minimum=100, value=1000, maximum=1000)
eighth_resolution_button = gr.Button("Eigth")
quarter_resolution_button = gr.Button("Quarter")
third_resolution_button = gr.Button("Third")
half_resolution_button = gr.Button("Half")
full_resolution_button = gr.Button("Full")

#   Image
sel_mask = gr.Image(label="Selected mask image", elem_id="stripper_sel_mask", elem_classes=["stripper-image"], type="numpy", tool="sketch", brush_radius=12, show_label=False, height=1000, interactive=True, visible=True)
mask_image = gr.Image(label="Mask", elem_classes=["stripper-image"], height=1000, visible=True)
add_mask_btn = gr.Button("Add mask by sketch", elem_id="stripper-add_mask_btn", elem_classes=["stripper-image"], visible=True)
trim_mask_btn = gr.Button("Trim mask by sketch", elem_id="stripper-trim_mask_btn", elem_classes=["stripper-image"], visible=True)
clear_mask_btn = gr.Button("Clear Mask", elem_id="stripper-clear-mask-btn", elem_classes=["stripper-image"], visible=True)
image_inpaint_button = gr.Button("Inpaint", elem_id="stripper-image_inpaint_btn", elem_classes=["stripper-video"])
#   Video
mask_video = gr.Video(label="Mask Video", elem_id="stripper-mask-video", elem_classes=["stripper-video"], source="upload", height=500, interactive=True, visible=False)
video_inpaint_button = gr.Button("Inpaint", visible=False)

#Outpaint
outpaint_up_slider = gr.Slider(label="Outpaint Up", value=0, maximum=1000, visible=False)
outpaint_down_slider = gr.Slider(label="Outpaint Down", value=0, maximum=1000, visible=False)
outpaint_left_slider = gr.Slider(label="Outpaint Left", value=0, maximum=1000, visible=False)
outpaint_right_slider = gr.Slider(label="Outpaint Right", value=0, maximum=1000, visible=False)
outpaint_button = gr.Button("Outpaint", visible=False)


prompt = gr.Textbox(label="Prompt", elem_id="stripper_sd_prompt")
n_prompt = gr.Textbox(label="Negative Prompt", elem_id="stripper_sd_n_prompt")
sampling_steps_slider = gr.Slider(label="Sampling Steps", elem_id="stripper_sampling_steps", minimum=1, maximum=100, value=40, step=1)
cfg_scale_slider = gr.Slider(label="Guidance Scale", elem_id="stripper_cfg_scale", minimum=0.1, maximum=30.0, value=4, step=0.1)
seed_slider = gr.Slider(label="Seed", elem_id="stripper_sd_seed", minimum=-1, maximum=2147483647, step=1, value=-1)

iteration_count_slider = gr.Slider(label="Iterations", elem_id="stripper_iteration_count_slider", minimum=1, maximum=100, value=2, step=1)

out_gallery_kwargs = dict(columns=2, object_fit="contain", height=1000, preview=True)
output_gallery = gr.Gallery(label="Output", kwargs=out_gallery_kwargs)
output_video = gr.Video(label="Output", kwargs=out_gallery_kwargs)


#Body
age_Slider = gr.Slider(label="Age", min=0, value=18, max=100, step=1)
body_type_dropdown = gr.Dropdown(label="Body Type", choices=BODY_TYPES, value="Slim")
neck_is_visible_check         = gr.Checkbox(label="Neck Is Visisble", value=True)
shoulders_are_visible_check   = gr.Checkbox(label="Shoulders Are Visisble", value=True)
arms_are_visible_check        = gr.Checkbox(label="Arms Are Visisble", value=True)
hands_are_visible_check       = gr.Checkbox(label="Hands Are Visisble", value=True)
collar_bone_is_visible_check  = gr.Checkbox(label="Collar Bone Is Visisble", value=True)
chest_is_visible_check        = gr.Checkbox(label="Chest Is Visisble", value=True)
back_is_visible_check         = gr.Checkbox(label="Back Is Visisble", value=True)
boobs_are_visible_check       = gr.Checkbox(label="Boobs Are Visisble", value=True)
boobs_size_dropdown           = gr.Dropdown(label="Boobs Size", choices=BOOB_SIZES, value=BOOB_SIZES[0])
nipples_are_visible_check     = gr.Checkbox(label="Nipples Are Visisble", value=True)
hard_nipples_check            = gr.Checkbox(label="Hard Nipples", value=True) 
waist_is_visible_check        = gr.Checkbox(label="Waist Is Visisble", value=True)
narrow_waist_check            = gr.Checkbox(label="Narrow Waist", value=True)

belly_is_visible_check        = gr.Checkbox(label="Belly Is Visisble", value=True)
belly_button_is_visible_check = gr.Checkbox(label="Belly Button Is Visisble", value=True)
abs_dropdown                  = gr.Dropdown(label="Abs", choices=ABS_OPTIONS, value=ABS_OPTIONS[0])
is_pregnant_check             = gr.Checkbox(label="Is Pregnant", value=False)

vagina_is_visible_check       = gr.Checkbox(label="Vagina Is Visisble", value=True)
pubic_hair_check              = gr.Checkbox(label="Pubic Hair", value=True)
ass_is_visible_check          = gr.Checkbox(label="Ass Is Visisble", value=True)
ass_size_dropdown             = gr.Dropdown(label="Ass Size", choices=ASS_SIZES, value=ASS_SIZES[0])
hips_are_visible_check        = gr.Checkbox(label="Hips Are Visisble", value=True)
thighs_are_visible_check      = gr.Checkbox(label="Thighs Are Visisble", value=True)
knees_are_visible_check       = gr.Checkbox(label="Knees Are Visisble", value=True)
shins_are_visible_check       = gr.Checkbox(label="Shins Are Visisble", value=True)
feet_are_visible_check        = gr.Checkbox(label="Feet Are Visisble", value=True)

generate_prompts_inputs = [age_Slider, body_type_dropdown, neck_is_visible_check, shoulders_are_visible_check, arms_are_visible_check, hands_are_visible_check, 
                                collar_bone_is_visible_check, chest_is_visible_check, back_is_visible_check, 
                                boobs_are_visible_check, boobs_size_dropdown, 
                                nipples_are_visible_check, hard_nipples_check, 
                                waist_is_visible_check, narrow_waist_check,
                                belly_is_visible_check, belly_button_is_visible_check, abs_dropdown, is_pregnant_check,
                                vagina_is_visible_check, pubic_hair_check,
                                ass_is_visible_check, ass_size_dropdown,
                                hips_are_visible_check,
                                thighs_are_visible_check, knees_are_visible_check, shins_are_visible_check, feet_are_visible_check]
generate_prompts_outputs = [prompt, n_prompt]

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        input_image_longest_side_resolution = gr.State(0)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    media_type_dropdown.render()
                    media_type_dropdown.change(fn=change_media_specific_visisbility, inputs=[media_type_dropdown, generation_type_dropdown], outputs=change_media_specific_visisbility_outputs)

                    generation_type_dropdown.render()
                    generation_type_dropdown.change(fn=change_media_specific_visisbility, inputs=[media_type_dropdown, generation_type_dropdown], outputs=change_media_specific_visisbility_outputs)

                input_image.render()

                input_video.render()
                with gr.Row():
                    with gr.Column():
                        sel_mask.render()
                        with gr.Row():
                            add_mask_btn.render()
                            add_mask_btn.click(add_mask, inputs=[input_image, sel_mask, mask_image], outputs=[sel_mask, mask_image]).then(None, inputs=None, outputs=None, _js="stripper_clearSelMask")
    
                            trim_mask_btn.render()
                            trim_mask_btn.click(trim_mask, inputs=[input_image, sel_mask, mask_image], outputs=[sel_mask, mask_image]).then(None, inputs=None, outputs=None, _js="stripper_clearSelMask")

                            clear_mask_btn.render()
                            clear_mask_btn.click(clear_mask, inputs=[input_image], outputs=[sel_mask, mask_image]).then(None, inputs=None, outputs=None, _js="stripper_clearSelMask")
                        
                        mask_image.render()
                        mask_image.upload(mask_uploaded, inputs=[input_image, mask_image], outputs=[sel_mask])

                        image_inpaint_button.render()
                        image_inpaint_button.click(generate_controlnet_inapint_images, inputs=[input_image, mask_image, pose_image, prompt, n_prompt, sampling_steps_slider, cfg_scale_slider, seed_slider, iteration_count_slider, inpaint_model_dropdown, inpaint_resolution_slider], outputs=[output_gallery])

                        mask_video.render()    

                        video_inpaint_button.render()                
                        video_inpaint_button.click(run_video_inpaint, inputs=[input_video, mask_video, prompt, n_prompt, sampling_steps_slider, cfg_scale_slider, seed_slider], outputs=[output_video])

                        outpaint_up_slider.render()
                        outpaint_down_slider.render()
                        outpaint_left_slider.render()
                        outpaint_right_slider.render() 

                        outpaint_button.render()
                        outpaint_button.click(generate_controlnet_outpaint_images, inputs=[input_image, outpaint_up_slider, outpaint_down_slider, outpaint_left_slider, outpaint_right_slider, prompt, n_prompt, sampling_steps_slider, cfg_scale_slider, seed_slider, iteration_count_slider, inpaint_model_dropdown, inpaint_resolution_slider], outputs=[output_gallery])
            with gr.Column():
                with gr.Accordion("Body"):
                    with gr.Row():
                        age_Slider.render()
                        age_Slider.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                        body_type_dropdown.render()
                        body_type_dropdown.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                    
                    neck_is_visible_check.render()
                    neck_is_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                    with gr.Row():
                        with gr.Accordion("Limbs"):
                            with gr.Accordion("Arms"):
                                shoulders_are_visible_check.render()
                                shoulders_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                arms_are_visible_check.render()
                                arms_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                                
                                hands_are_visible_check.render()
                                hands_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                            with gr.Accordion("Legs"):
                                hips_are_visible_check.render()
                                hips_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                thighs_are_visible_check.render()
                                thighs_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                knees_are_visible_check.render()
                                knees_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                shins_are_visible_check.render()
                                shins_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                feet_are_visible_check.render()
                                feet_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                        with gr.Accordion("Torso"):
                            with gr.Accordion("Waist"):
                                waist_is_visible_check.render()
                                waist_is_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                narrow_waist_check.render()
                                narrow_waist_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                            with gr.Accordion("Front"):
                                collar_bone_is_visible_check.render()
                                collar_bone_is_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                                with gr.Accordion("Chest"):
                                    chest_is_visible_check.render()
                                    chest_is_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                    with gr.Accordion("Boobs"):
                                        boobs_are_visible_check.render()
                                        boobs_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                        boobs_size_dropdown.render()
                                        boobs_size_dropdown.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                        with gr.Accordion("Nipples"):
                                            nipples_are_visible_check.render()
                                            nipples_are_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                            hard_nipples_check.render()
                                            hard_nipples_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                                with gr.Accordion("Belly"):
                                    with gr.Row():
                                        with gr.Column():
                                            belly_is_visible_check.render()
                                            belly_is_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                            belly_button_is_visible_check.render()
                                            belly_button_is_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                                        with gr.Column():
                                            abs_dropdown.render()
                                            abs_dropdown.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                            is_pregnant_check.render()            
                                            is_pregnant_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                                with gr.Accordion("Vagina"):
                                    vagina_is_visible_check.render()
                                    vagina_is_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                    pubic_hair_check.render()
                                    pubic_hair_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                            with gr.Accordion("Back"):
                                back_is_visible_check.render()
                                back_is_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                with gr.Accordion("Ass"):
                                    ass_is_visible_check.render()
                                    ass_is_visible_check.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)

                                    ass_size_dropdown.render()
                                    ass_size_dropdown.change(generate_stripper_prompts, inputs=generate_prompts_inputs, outputs=generate_prompts_outputs)
                with gr.Accordion("Prompts"):
                    with gr.Row():
                        prompt.render()
                        n_prompt.render()
                with gr.Accordion("Generation Settings"):
                    inpaint_resolution_slider.render()
                    with gr.Row():
                        eighth_resolution_button.render()
                        eighth_resolution_button.click(fn=lambda x: x // 8, inputs=[input_image_longest_side_resolution], outputs=[inpaint_resolution_slider])
                        
                        quarter_resolution_button.render()
                        quarter_resolution_button.click(fn=lambda x: x // 4, inputs=[input_image_longest_side_resolution], outputs=[inpaint_resolution_slider])

                        third_resolution_button.render()
                        third_resolution_button.click(fn=lambda x: x // 3, inputs=[input_image_longest_side_resolution], outputs=[inpaint_resolution_slider])

                        half_resolution_button.render()
                        half_resolution_button.click(fn=lambda x: x // 2, inputs=[input_image_longest_side_resolution], outputs=[inpaint_resolution_slider])

                        full_resolution_button.render()
                        full_resolution_button.click(fn=lambda x: x, inputs=[input_image_longest_side_resolution], outputs=[inpaint_resolution_slider])
                    sampling_steps_slider.render()
                    with gr.Row():
                        cfg_scale_slider.render()
                        iteration_count_slider.render()
                    inpaint_model_dropdown.render()
                    seed_slider.render()
                    with gr.Accordion("Controlnet", open=False):
                        pose_image.render()

        output_gallery.render()
        output_video.render()

        input_image.upload(input_image_upload, inputs=[input_image], outputs=[sel_mask, mask_image, pose_image, inpaint_resolution_slider, input_image_longest_side_resolution])

    return [(ui_component, "Stripper", "stripper")]

script_callbacks.on_ui_tabs(on_ui_tabs)

change_media_specific_visisbility_outputs = [input_image, sel_mask, add_mask_btn, trim_mask_btn, clear_mask_btn, mask_image, image_inpaint_button, input_video, mask_video, video_inpaint_button, outpaint_up_slider, outpaint_down_slider, outpaint_left_slider, outpaint_right_slider, outpaint_button]
def change_media_specific_visisbility(media_type:str, generation_type:str):
    is_image_mode = media_type == "Image"
    is_video_mode = media_type == "Video"

    is_inpaint_mode = generation_type == "Inpaint"
    is_outpaint_mode = generation_type == "Outpaint"

    return (gr.update(visible=is_image_mode), gr.update(visible=is_image_mode and is_inpaint_mode), gr.update(visible=is_image_mode and is_inpaint_mode), gr.update(visible=is_image_mode and is_inpaint_mode),  gr.update(visible=is_image_mode and is_inpaint_mode), gr.update(visible=is_image_mode and is_inpaint_mode), gr.update(visible=is_image_mode and is_inpaint_mode), gr.update(visible=is_video_mode), gr.update(visible=is_video_mode and is_inpaint_mode), gr.update(visible=is_video_mode and is_inpaint_mode),
        gr.update(visible=is_outpaint_mode), gr.update(visible=is_outpaint_mode), gr.update(visible=is_outpaint_mode), gr.update(visible=is_outpaint_mode), gr.update(visible=is_outpaint_mode))

def input_image_upload(input_image: np.ndarray):
    height, width, channels = input_image.shape

    longest_side_resolution:int = height if height > width else width 
    scaled_input_image = scale_image(input_image)

    default_resolution = longest_side_resolution if longest_side_resolution < 1000 else 1000

    mask_image = Image.new("RGB", (width, height), "black")
    
    pose_image = get_pose_image(input_image)

    return gr.Image.update(value=scaled_input_image), gr.Image.update(value=mask_image), gr.Image.update(value=pose_image), gr.Slider.update(value=default_resolution, maximum=longest_side_resolution), longest_side_resolution

def get_pose_image(input_image: np.ndarray):
    openpose_image = openpose(np.array(input_image, np.uint8), hand_and_face=True)

    return openpose_image

#Mask functions
def mask_uploaded(input_image, mask_image):
    scaled_input_image = scale_image(input_image)
    height, width, channels = scaled_input_image.shape

    scaled_mask_image = cv2.resize(mask_image, (width, height), interpolation=cv2.INTER_LINEAR)
    compisite_image = cv2.addWeighted(scaled_input_image, 0.5, scaled_mask_image, 0.5, 0)

    return compisite_image


def add_mask(input_image, sel_mask, mask_image):
    mask_height, mask_width = mask_image.shape[:2]

    previous_selection_mask = sel_mask["mask"][:, :, 0:3].astype(bool).astype(np.uint8)
    composite_height, composite_width = previous_selection_mask.shape[:2]

    scaled_up_selection_mask = cv2.resize(previous_selection_mask, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)
    scaled_down_input_image = cv2.resize(input_image, (composite_width, composite_height), interpolation=cv2.INTER_LINEAR)

    new_mask = mask_image + (scaled_up_selection_mask * np.invert(mask_image, dtype=np.uint8))
    scale_down_new_mask = cv2.resize(new_mask, (composite_width, composite_height), interpolation=cv2.INTER_LINEAR)

    compisite_image = cv2.addWeighted(scaled_down_input_image, 0.5, scale_down_new_mask, 0.5, 0)

    return compisite_image, new_mask

def trim_mask(input_image, sel_mask, mask_image):
    mask_height, mask_width = mask_image.shape[:2]

    previous_selection_mask = np.logical_not(sel_mask["mask"][:, :, 0:3].astype(bool)).astype(np.uint8)
    composite_height, composite_width = previous_selection_mask.shape[:2]

    scaled_up_selection_mask = cv2.resize(previous_selection_mask, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)
    scaled_down_input_image = cv2.resize(input_image, (composite_width, composite_height), interpolation=cv2.INTER_LINEAR)

    new_mask = mask_image * scaled_up_selection_mask
    scale_down_new_mask = cv2.resize(new_mask, (composite_width, composite_height), interpolation=cv2.INTER_LINEAR)

    compisite_image = cv2.addWeighted(scaled_down_input_image, 0.5, scale_down_new_mask, 0.5, 0)

    return compisite_image, new_mask

def clear_mask(input_image):
    height, width, channels = input_image.shape

    scaled_input_image = scale_image(input_image)
    
    mask_image = Image.new("RGB", (width, height), "black")

    return scaled_input_image, mask_image

def run_video_inpaint(video_path:str, mask_path:str,
                      prompt:str, n_prompt:str,
                      sampling_steps:str, cfg_scale:float, seed:int=-1, inpaint_model_id:str="Uminosachi/realisticVisionV51_v51VAE-inpainting", 
                      iteration_count:int=1):
    #Video Capture
    video = cv2.VideoCapture(video_path)
    mask = cv2.VideoCapture(mask_path)
    fps:int = video.get(cv2.CAP_PROP_FPS)

    sampler_name:str = "DPM2 a Karras"

    output_folder:str = get_save_folder()

    if seed == -1:
        seed = random.randint(0, 2147483647)

    #Get video Frames
    video_frames: list = []
    mask_frames:list = []

    while True:
        video_ret, video_frame = video.read()
        mask_ret, mask_frame = mask.read()

        if not video_ret and not mask_ret:
            break

        if video_ret:
            video_frames.append(video_frame)

        if mask_ret:
            mask_frames.append(mask_frame)

        

    original_height, original_width = video_frames[0].shape[:2]

    if len(video_frames) != len(mask_frames):
        print(f"video frame count ({len(video_frames)}) does not match mask frame count ({len(mask_frames)})")
        return []

    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipe, torch_generator = create_inpaint_pipline(inpaint_model_id, sampler_name)

    output_video_path:str = f"{output_folder}/composite.mp4"

    #Video Saving
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    output_video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))

    pbar = tqdm(total=len(video_frames), unit="frame", unit_scale=True)
    for frame_index in range(len(video_frames)):
        current_frame = video_frames[frame_index]
        current_mask = mask_frames[frame_index]

        input_image = scale_image(current_frame)
        mask_image = scale_image(current_mask)


        init_image, mask_image = auto_resize_to_pil([input_image, mask_image])
        width, height = init_image.size

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
        composite_image = combine_images(current_frame, output_image, current_mask)
        #composite_image.save(f"/mnt/327939d0-7bdf-483f-bbc4-c54500c75606/Virtual Machines/Share Folder/{frame_index}.jpg")
        composite_image_numpy = np.array(composite_image)

        output_video_writer.write(composite_image_numpy)
        pbar.update(1)

    output_video_writer.release()

    return output_video_path

def get_save_folder() -> str:
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    save_folder_path = f"{output_folder}/{formatted_time}"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    return save_folder_path

