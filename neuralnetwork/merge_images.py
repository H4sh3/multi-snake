import os
import numpy as np
from PIL import Image

def remove_black_background(image):
    """
    Remove black background from an image.
    
    Args:
    image (PIL.Image): Input image
    
    Returns:
    PIL.Image: Image with black background removed
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create a mask for non-black pixels
    # Threshold can be adjusted (currently removes pure black)
    mask = (img_array[:,:,:3] > [10, 10, 10]).any(axis=2)
    
    # Convert mask to 4 channel image (RGBA)
    img_array_copy = img_array.copy()
    img_array_copy[~mask] = [0, 0, 0, 0]
    
    return Image.fromarray(img_array_copy)

def overlay_images(folder_path, output_path='overlay_result.png'):
    """
    Overlay all images in a specified folder, removing black backgrounds.
    
    Args:
    folder_path (str): Path to the folder containing images
    output_path (str): Path to save the final overlaid image
    """
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Sort files to ensure consistent order
    image_files.sort()
    
    # Open the first image as the base
    if not image_files:
        raise ValueError("No image files found in the specified folder")
    
    base_image = Image.open(os.path.join(folder_path, image_files[0])).convert("RGBA")
    base_image = remove_black_background(base_image)
    
    # Overlay subsequent images
    for image_file in image_files[1:]:
        img_path = os.path.join(folder_path, image_file)
        overlay = Image.open(img_path).convert("RGBA")
        overlay = remove_black_background(overlay)
        
        # Resize overlay to match base image if needed
        if overlay.size != base_image.size:
            overlay = overlay.resize(base_image.size)
        
        # Blend images
        base_image = Image.alpha_composite(base_image, overlay)
    
    # Save the final image
    base_image.save(output_path)
    print(f"Overlaid image saved to {output_path}")

# Example usage
overlay_images('images')