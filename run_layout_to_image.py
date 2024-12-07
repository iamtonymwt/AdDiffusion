import os
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from utils.generation_utils import load_checkpoint, bbox_encode, draw_layout

########################
# Set random seed
#########################
from accelerate.utils import set_seed
set_seed(0)

def run_layout_to_image(layout, args):
  ########################
  # Build pipeline
  #########################
  pipe, generation_config = load_checkpoint(args.ckpt_path)
  pipe = pipe.to("cuda")
  args = {arg: getattr(args, arg) for arg in vars(args) if getattr(args, arg) is not None}
  generation_config.update(args)
  
  # Sometimes the nsfw checker is confused by the Pokémon images, you can disable
  # it at your own risk here
  disable_safety = True
  if disable_safety:
    def null_safety(images, **kwargs):
        return images, False
    pipe.safety_checker = null_safety
  

  assert not generation_config['dataset'] == 'nuimages' or "timeofday" not in layout or layout['timeofday'] in ['daytime', 'night']
  assert not generation_config['dataset'] == 'nuimages' or "weather" not in layout or layout['weather'] in ['sunny', 'rain']
  if "timeofday" in generation_config['prompt_template'] and "timeofday" not in layout.keys():
    layout["timeofday"] = "daytime"
  if "weather" in generation_config['prompt_template'] and "weather" not in layout.keys():
    layout["weather"] = "sunny"

  # camera sanity check
  assert not generation_config['dataset'] == 'nuimages' or ("camera" in layout and layout['camera'] in ['front', 'front left', 'front right', 'back', 'back left', 'back right'])
  bboxes = layout['bbox'].copy()
  layout["bbox"] = bbox_encode(layout['bbox'], generation_config)
  prompt = generation_config['prompt_template'].format(**layout)
  print(prompt)
  
  ########################
  # Generation
  ########################
  # generation params
  width = generation_config["width"]  
  height = generation_config["height"]
  scale = generation_config["cfg_scale"]
  n_samples = generation_config["nsamples"]
  num_inference_steps = generation_config["num_inference_steps"]
  
  # run generation
  images = pipe(n_samples*[prompt], guidance_scale=scale, num_inference_steps=num_inference_steps, height=int(height), width=int(width)).images
  
  ########################
  # Save results
  #########################
  root = args["output_dir"]
  os.makedirs(root, exist_ok=True)
  layout_canvas = draw_layout(bboxes)
  layout_canvas = Image.fromarray(layout_canvas, mode='RGB').save(os.path.join(root, '{}_layout.jpg'.format(generation_config['dataset'])))
  for idx, image in enumerate(images):
    image = np.asarray(image)
    image = Image.fromarray(image, mode='RGB')
    image = image.resize((1600, 900))
    image.save(os.path.join(root, '{}_{}.jpg'.format(generation_config['dataset'], idx)))

if __name__ == "__main__":
  parser = ArgumentParser(description='Layout-to-image generation script')
  parser.add_argument('ckpt_path', type=str)
  parser.add_argument('--nsamples', type=int, default=50)
  parser.add_argument('--cfg_scale', type=float, default=None)
  parser.add_argument('--num_inference_steps', type=int, default=None)
  parser.add_argument('--output_dir', type=str, default="./results/")
  args = parser.parse_args()
  
  ########################
  # Define layouts
  # Note: 
  # 1) "camera": specific for nuimages, and should be selected from [front, front left, front right, back, back left, back right]
  # 2) "bbox": list of bounding boxes, each defined as [category, x1, y1, x2, y2] 
  #   a) category (str):, check dataset2classes in utils.generation_utils
  #   b) x1, y1, x2, y2 (float): in range of [0, 1]
  ########################
  
  ########################
  layout = {
    "camera": "back",
    "timeofday": "night",
    "weather": "rain",
    "bbox": [
      ["car", 0.76625, 0.5277777777777778, 0.88375, 0.6288888888888889],
      ["car", 0.51125, 0.5066666666666667, 0.575625, 0.5933333333333334],
      ["car", 0.614375, 0.5133333333333333, 0.675625, 0.5844444444444444],
      ["car", 0.4075, 0.5177777777777778, 0.460625, 0.5833333333333334],
      ["car", 0.57, 0.5088888888888888, 0.61625, 0.5711111111111111],
    ]
  }
  
  ########################
  # Run layout-to-image generation
  ########################
  run_layout_to_image(layout, args)