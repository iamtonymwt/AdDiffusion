#############
# feature-based CLIP image similarity
#############
import os
import clip
import torch
from PIL import Image

# Load the CLIP model and specify the device (use "cuda" for GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image_similarity = []
text_similarity = []

gt_image_path = "/home/xiao/AdDiffusion/results/road1.png"
test_image_directory = "/home/xiao/AdDiffusion/results/"
# camera = "front"
# timeofday = "daytime"
# weather = "sunny"
# location = "boston"
# gt_prompt = f"A {timeofday} {weather} driving scene in {location}"
# gt_prompt = f"road"

# Load and preprocess two images
image_gt = preprocess(Image.open(gt_image_path)).unsqueeze(0).to(device)  # Move image to GPU

for filename in os.listdir(test_image_directory):
    file_path = os.path.join(test_image_directory, filename)
    image_generated = preprocess(Image.open(file_path)).unsqueeze(0).to(device)  # Move image to GPU
    # Tokenize text
    # text = clip.tokenize([gt_prompt]).to(device)
    
    # Extract features for both images
    with torch.no_grad():
        features1 = model.encode_image(image_gt)  # GPU accelerated
        features2 = model.encode_image(image_generated)  # GPU accelerated
        # text_features = model.encode_text(text)

    # Normalize the features
    features1 /= features1.norm(dim=-1, keepdim=True)
    features2 /= features2.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = (features1 @ features2.T).item()
    print(f"Cosine Similarity (GPU accelerated): {similarity}")
    image_similarity.append(similarity)

    # similarity2 = (features2 @ text_features.T).item()
    # print(f"CLIP Similarity Score: {similarity2}")
    # text_similarity.append(similarity2)
    
average_image_similarity = sum(image_similarity)/len(image_similarity)
# average_text_similarity = sum(text_similarity)/len(text_similarity)

print("average_image_similarity is:", average_image_similarity)
# print("average_text_similarity is:", average_text_similarity)

# #############
# # feature-based CLIP text-image silimarity
# #############
# import torch
# import clip
# from PIL import Image

# # Load the model and tokenizer
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# # Example: Load an image and its corresponding description
# image_path = "./generated/test_bbox_generated_retangle.jpeg"
# text_description = "A daytime sunny driving scene image of front camera"

# # Preprocess image
# image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)



# # Get embeddings
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)

# # Normalize embeddings
# image_features = image_features / image_features.norm(dim=-1, keepdim=True)
# text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# # Compute cosine similarity
# similarity = (image_features @ text_features.T).item()
# print(f"CLIP Similarity Score: {similarity}")


