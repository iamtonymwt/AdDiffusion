import os
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from utils.generation_utils import load_checkpoint, bbox_encode, draw_layout

def randomize_bbox():
    """Randomly initialize bbox positions and quantities."""
    num_bboxes = random.randint(0, 5)  # Randomize number of bounding boxes
    bboxes = []
    for _ in range(num_bboxes):
        category = "car"
        x1, y1 = random.uniform(0, 0.7), random.uniform(0, 0.6)
        x2, y2 = x1 + random.uniform(0.2, 0.3), y1 + random.uniform(0.3, 0.4)
        x2, y2 = min(x2, 1.0), min(y2, 1.0)  # Ensure coordinates are within bounds
        bboxes.append([category, x1, y1, x2, y2])
    return bboxes

def run_layout_to_image(pipe, generation_config, layout, args, iteration):
    ########################
    # Encode layout and build text prompt
    #########################
    bboxes = layout['bbox'].copy()
    layout["bbox"] = bbox_encode(layout['bbox'], generation_config)
    prompt = generation_config['prompt_template'].format(**layout)
    print(f"Prompt for iteration {iteration}: {prompt}")

    ########################
    # Generation
    ########################
    width = generation_config["width"]
    height = generation_config["height"]
    scale = generation_config["cfg_scale"]
    n_samples = args.nsamples
    num_inference_steps = generation_config["num_inference_steps"]

    images = pipe(n_samples * [prompt], guidance_scale=scale, num_inference_steps=num_inference_steps, height=int(height), width=int(width)).images

    ########################
    # Save results
    #########################
    root = args.output_dir
    os.makedirs(root, exist_ok=True)

    # Save layout as image
    layout_canvas = draw_layout(bboxes)
    layout_canvas = Image.fromarray(layout_canvas, mode='RGB')
    # layout_canvas.save(os.path.join(root, f"layout_{iteration}.jpg"))

    # Save generated images with resizing to 1600x900
    for idx, image in enumerate(images):
        image = np.asarray(image)
        image = Image.fromarray(image, mode='RGB')
        image = image.resize((1600, 900), Image.Resampling.LANCZOS)
        image.save(os.path.join(root, f"image_{iteration}_{idx}.jpg"))

if __name__ == "__main__":
    parser = ArgumentParser(description='Layout-to-image generation script')
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--cfg_scale', type=float, default=None)
    parser.add_argument('--num_inference_steps', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default="./results/")
    parser.add_argument('--num_iterations', type=int, default=60, help='Number of random layouts to generate')
    args = parser.parse_args()

    ########################
    # Load model once
    ########################
    pipe, generation_config = load_checkpoint(args.ckpt_path)
    pipe = pipe.to("cuda")

    # Disable safety checker
    disable_safety = True
    if disable_safety:
        def null_safety(images, **kwargs):
            return images, False
        pipe.safety_checker = null_safety

    ########################
    # Define base layout
    ########################
    base_layout = {
        "camera": "front",
        "timeofday": "night",
        "weather": "sunny",
        "bbox": []  # This will be randomized
    }

    ########################
    # Generate multiple layouts
    ########################
    print(generation_config)
    for iteration in range(args.num_iterations):
        base_layout['bbox'] = randomize_bbox()
        run_layout_to_image(pipe, generation_config, base_layout, args, iteration)
