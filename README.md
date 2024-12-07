## Env
```
pip install -r ./requirements.txt
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
mim install mmcv-full==1.7.2
```

## Dataset
### dataset preparation (processed NuScenes)
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

### file names of images in the dataset 
```
samples/{cam_pos}/{info}__{cam_pos}__{timestamp}__{weather)__{light}__{location}.jpg
```

## test
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


## train
1. update reading path 
    ```
    anaconda3/envs/geodiffusion/lib/python3.9/site-packages/mmcv/fileio/file_client.py
    line538: filepath = '__'.join(filepath.split('__')[:3])+'.jpg'
    ```
2. bash 
    ```
    bash /AdDiffusion/tools/dist_train.sh
    ```