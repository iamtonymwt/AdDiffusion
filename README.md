# Self Driving-Oriented Image Generation With Condition Control based on Stable Diffusion

Code base for CSC2541 course project by Sean, Summer, and Tony @ UofT

-- A Fully conditioned autonoumous driving scene generation model based on stable diffusion

<div align="center">
<img src="assets/conditioning-diagram.png" alt="" width="1000">
support multi conditions, including weather, light, region, camera view, and car box
</div>

## Model Explaination
<div align="center">
<img src="assets/method-diagram.png" alt="" width="1000">
The overall structure of our model. We adopted the pipeline of Stable Diffusion as GeoDiffusion did so. Besides adding weather, lighting and location for controlled image generation, we replaced the vanilla attention inside denoising U-Net with Differential Attention. We fine-tuned the GeoDiffusion model by feeding in new text prompts for cross-attention (while it can also act as part of input as the 'Switch' suggests). During fine-tuning, only the parameters in text encoder and denoising U-Net were updated, whereas the VQ-VAE model was frozen.
</div>



## Differential Attention Implementation
in /anaconda3/envs/geodiffusion/lib/python3.9/site-packages/diffusers/models/attention_processor.py
1. Adjust the 'Attention' Class 
    ```
    self.head_dim = self.inner_dim // self.heads // 2
    self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * 8)
    self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
    self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
    self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
    self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
    ```
2. Adjust the 'AttnProcesser2_0' Class
    ```
    # torch.Size([2, 8, 4096, 40])
    q1 = query[:, :, :, 0:head_dim//2]
    q2 = query[:, :, :, head_dim//2:]
    k1 = key[:, :, :, 0:head_dim//2]
    k2 = key[:, :, :, head_dim//2:]

    # calculate two attention maps
    attention_score1 = torch.matmul(q1, k1.transpose(-1, -2))
    attention_score2 = torch.matmul(q2, k2.transpose(-1, -2))

    attention_score1 = attention_score1 / math.sqrt(head_dim)
    attention_score2 = attention_score2 / math.sqrt(head_dim)

    attention_probs1 = nn.functional.softmax(attention_score1, dim=-1)
    attention_probs2 = nn.functional.softmax(attention_score2, dim=-1)
    

    # calculate the difference
    lambda_1 = torch.exp(torch.sum(attn.lambda_q1 * attn.lambda_k1, dim=-1).float()).type_as(q1)
    lambda_2 = torch.exp(torch.sum(attn.lambda_q2 * attn.lambda_k2, dim=-1).float()).type_as(q1)
    lambda_full = lambda_1 - lambda_2 + attn.lambda_init

    attention_probs = attention_probs1 - lambda_full * attention_probs2
    
    # Mask heads if we want to
    if attention_mask is not None:
        attention_probs = attention_probs * attention_mask

    hidden_states = torch.matmul(attention_probs, value)
    ```


## Env Preparation
```
pip install -r ./requirements.txt
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
mim install mmcv-full==1.7.2
```

## Dataset Preparation
### Downloading Dataset (processed NuScenes)
1. Download annotations file and unzip from [TODO]
2. Create following folders under data/nuimages/samples:
    ```
    CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT
    ```
3. download following image files
    ```
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval01_keyframes.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval02_keyframes.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_keyframes.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval04_keyframes.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_keyframes.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_keyframes.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_keyframes.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval08_keyframes.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval09_keyframes.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval10_keyframes.tgz
    ```
4. unzip and move the images to the folders we created in step 2

### File names of images in the dataset 
```
samples/{cam_pos}/{info}__{cam_pos}__{timestamp}__{weather)__{light}__{location}.jpg
```

## Run Training
1. update reading path 
    ```
    anaconda3/envs/geodiffusion/lib/python3.9/site-packages/mmcv/fileio/file_client.py
    line538: filepath = '__'.join(filepath.split('__')[:3])+'.jpg'
    ```
2. bash 
    ```
    bash /AdDiffusion/tools/dist_train.sh
    ```
3. Check Tensorboard
    ```
    tensorboard --logdir [work_dir]
    ```


## Run Testing
1. download pretrained ckpt
    ```
    sudo apt install git-lfs
    git clone https://huggingface.co/KaiChen1998/geodiffusion-nuimages-time-weather-512x512
    cd geodiffusion-nuimages-time-weather-512x512
    git lfs install
    git lfs pull
    ```
2. test
    ```
    python run_layout_to_image.py $CKPT_PATH --output_dir ./results/
    ```

## Run Evaluation
1. install clip
    ```
    pip install git+https://github.com/openai/CLIP.git
    ```
2. Test CLIP Similarity
    ```
    python /AdDiffusion/eval/clip_similarity.py
    ```
3. Test FID
    ```
    python /AdDiffusion/eval/fid.py
    ```

## Demos
<div align="center">
<img src="assets/visualizaton1.png" alt="" width="1000">
<img src="assets/visualizaton2.png" alt="" width="1000">
</div>
