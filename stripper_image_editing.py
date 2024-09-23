from torchvision import transforms

import cv2
import numpy as np

from PIL import Image, ImageFilter, ImageDraw

def combine_images(inpaint_input_image, inpaint_image, inpaint_mask) -> Image.Image:
    if isinstance(inpaint_input_image, np.ndarray):
        inpaint_input_image = Image.fromarray(inpaint_input_image, 'RGB')

    if isinstance(inpaint_mask, np.ndarray):
        inpaint_mask = Image.fromarray(inpaint_mask, 'RGB')  # Ensure mask is in grayscale mode

    width, height = inpaint_input_image.size

    #Scale inpaint image and mask up to original resolution
    inpaint_image = inpaint_image.resize((width, height))
    inpaint_mask = inpaint_mask.resize((width, height))

    dilate_mask_image = Image.fromarray(cv2.dilate(np.array(inpaint_mask), np.ones((3, 3), dtype=np.uint8), iterations=4))
    output_image = Image.composite(inpaint_image, inpaint_input_image, dilate_mask_image.convert("L").filter(ImageFilter.GaussianBlur(3)))

    return output_image
def scale_image(image, target_resolution:int = 1000):
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.uint8)

    original_height, original_width = image.shape[:2]

    # Determine if width or height is the limiting factor
    if original_width >= original_height:
        # Scale based on width
        scale_factor = target_resolution / original_width
    else:
        # Scale based on height
        scale_factor = target_resolution / original_height        

    # Calculate the new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)


    #print(f"scaling image to {new_width} x {new_height}")
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return image

def auto_resize_to_pil(images):
    outputs = []
    for image in images:
        if type(image) == np.ndarray:
            image = Image.fromarray(image)

            width, height = image.size
            new_height = (height // 8) * 8
            new_width = (width // 8) * 8

            if new_width < width or new_height < height:
                if (new_width / width) < (new_height / height):
                    scale = new_height / height
                else:
                    scale = new_width / width
                resize_height = int(height*scale+0.5)
                resize_width = int(width*scale+0.5)
                if height != resize_height or width != resize_width:
                    image = transforms.functional.resize(image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
                if resize_height != new_height or resize_width != new_width:
                    image = transforms.functional.center_crop(image, (new_height, new_width))

            outputs.append(image)
    return outputs

def create_borderd_image(image, outpaint_up:int, outpaint_down:int, outpaint_left:int, outpaint_right:int, overlap:int = 10):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image, 'RGB')

    # Calculate new size
    new_width = image.width + outpaint_left + outpaint_right
    new_height = image.height + outpaint_up + outpaint_down

    # Create a new image with the new size
    new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    
    # Paste the original image onto the new image, centered within the new borders
    new_image.paste(image, (outpaint_left, outpaint_up))

    # Create a mask with the same size as the new image
    mask = Image.new("L", (new_width, new_height), 0)
    draw = ImageDraw.Draw(mask)

    # Add black borders and update the mask with overlap
    if outpaint_up > 0:
        black_border_top = Image.new("RGB", (new_width, outpaint_up), (0, 0, 0))
        new_image.paste(black_border_top, (0, 0))
        draw.rectangle([0, 0, new_width, outpaint_up + overlap], fill=255)
    
    if outpaint_down > 0:
        black_border_bottom = Image.new("RGB", (new_width, outpaint_down), (0, 0, 0))
        new_image.paste(black_border_bottom, (0, new_height - outpaint_down))
        draw.rectangle([0, new_height - outpaint_down - overlap, new_width, new_height], fill=255)
    
    if outpaint_left > 0:
        black_border_left = Image.new("RGB", (outpaint_left, new_height), (0, 0, 0))
        new_image.paste(black_border_left, (0, 0))
        draw.rectangle([0, 0, outpaint_left + overlap, new_height], fill=255)
    
    if outpaint_right > 0:
        black_border_right = Image.new("RGB", (outpaint_right, new_height), (0, 0, 0))
        new_image.paste(black_border_right, (new_width - outpaint_right, 0))
        draw.rectangle([new_width - outpaint_right - overlap, 0, new_width, new_height], fill=255)
    
    return new_image, mask
