import os
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from utils.generation_utils import load_checkpoint, bbox_encode, draw_layout

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

########################
# Set random seed
#########################
from accelerate.utils import set_seed
# set_seed(0)

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
  
  ########################
  # Encode layout and build text prompt
  #########################
  # timeofday and weather sanity check  
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
    print('before resize:', image.size)
    image = image.resize((1600, 900), Image.Resampling.LANCZOS)
    print('after resize:', image.size)
    image.save(os.path.join(root, '{}_{}.jpg'.format(generation_config['dataset'], idx)))

if __name__ == "__main__":
  parser = ArgumentParser(description='Layout-to-image generation script')
  parser.add_argument('ckpt_path', type=str)
  parser.add_argument('--nsamples', type=int, default=1)
  parser.add_argument('--cfg_scale', type=float, default=None)
  parser.add_argument('--num_inference_steps', type=int, default=None)
  parser.add_argument('--output_dir', type=str, default="./results/")
  parser.add_argument('--gt_image', type=str, default="./generated_result/Front_View/gt_sunny_daytime/n003-2018-01-02-11-48-43+0800__CAM_FRONT__1514865170978337.jpg")
  args = parser.parse_args()


  ########################
  # detect bbox from each trainset image
  # return normalized bbox as conditioning for generation
  ########################
  # Load a pre-trained YOLOv8 model
  model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt', 'yolov8m.pt', etc., for larger models

  # Load the input image
  image_path = args.gt_image  #   Replace with your image path
  image = cv2.imread(image_path)  # Read the image using OpenCV
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB for display

  height, width, channels = image.shape

  # Perform inference
  results = model(image_rgb)  # Run YOLO on the image

  # Filter results to detect cars (COCO class ID for cars is 2)
  car_boxes = []
  for result in results:
      for box in result.boxes:
          cls = int(box.cls)  # Class ID
          if cls == 2:  # Class ID 2 corresponds to cars
              bbox = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
              car_boxes.append(bbox)

  # Print the bounding boxes for cars
  print("Detected car bounding boxes:", car_boxes)

  normalized_boxes = []
  for box in car_boxes:
      x_min, y_min, x_max, y_max = map(int, box)
      x_min = x_min / width
      y_min = y_min / height
      x_max = x_max / width
      y_max = y_max / height
      normalized_boxes.append([x_min, y_min, x_max, y_max])

  print('Normarlized boxes:', normalized_boxes)
  
  ########################
  # Define layouts
  # Note: 
  # 1) "camera": specific for nuimages, and should be selected from [front, front left, front right, back, back left, back right]
  # 2) "bbox": list of bounding boxes, each defined as [category, x1, y1, x2, y2] 
  #   a) category (str):, check dataset2classes in utils.generation_utils
  #   b) x1, y1, x2, y2 (float): in range of [0, 1]
  ########################
  # example layout for nuimages
  formated_bbox = []
  for bbox in normalized_boxes:
      formated_bbox.append(["car", bbox[0], bbox[1], bbox[2], bbox[3]])
    
  layout = {
    "camera": "front",
    "timeofday": "night",
    "weather": "rain",
    "bbox": formated_bbox,
  }
  
  
  ########################
  # Run layout-to-image generation
  ########################
  run_layout_to_image(layout, args)