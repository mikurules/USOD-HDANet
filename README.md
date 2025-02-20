# HDANet: Enhancing Underwater Salient Object Detection with Physics-Inspired Multimodal Joint Learning

Result and source code for our paper “**HDANet: Enhancing Underwater Salient Object Detection with Physics-Inspired Multimodal Joint Learning**”.

## Method
The HDANet addresses USOD challenges through developing targeted designs. It first integrates a task-driven underwater image enhancement module, named HydroDepthEnhanceModule (HDEM). Furthermore, we develop a physics-inspired three-way unsupervised learning strategy, leveraging the complementary effects of re-enhancement and re-degradation to improve HDEM’s generalization across diverse underwater image degradation scenarios. Additionally, we design a robust cross-attention (RCA) module to effectively fuse multimodal features while mitigating noise and blurring by exploiting channel and spatial cross-attention mechanisms. 

<img src="fig1.png" alt="fig1" style="zoom:25%;" />

## Result

(1) **Trained models** are available [BaiduNetdisk](https://pan.baidu.com/s/1nndmH18X_3c_PjJXLgSysw?pwd=USOD) **fetch code**: [USOD]  &&& [Googledriven](https://drive.google.com/drive/folders/1kDGGTYmwsDH3RCFBUFBJZDYUxpPpu4G4?usp=drive_link) 

(2) **Predicted saliency maps of USOD10K** are available [BaiduNetdisk](https://pan.baidu.com/s/1vV_ire7XziNdCtUe8E6U6w?pwd=USOD) **fetch code**: [USOD] &&& [Google driven](https://drive.google.com/drive/folders/16hrFMeNGnyWfdw2_14PZM0Rluivhoc3J?usp=drive_link)

(3) **Predicted saliency maps of USOD** are available [BaiduNetdisk](https://pan.baidu.com/s/1vV_ire7XziNdCtUe8E6U6w?pwd=USOD) **fetch code**: [USOD]  &&& [Google driven](https://drive.google.com/drive/folders/16hrFMeNGnyWfdw2_14PZM0Rluivhoc3J?usp=drive_link)

## USOD dataset

USOD10K dataset:  Baidu Netdisk: [USOD10K](https://pan.baidu.com/s/1edg2B9HjnHdEpmwnUOT0-w) **fetch code**: [good]  &&&  Google drive: [USOD10K](https://drive.google.com/file/d/1PH0PwKchXnkWwtAwbhNSW4utMCp5zer8/view?usp=sharing).

USOD dataset:[USOD](https://irvlab.cs.umn.edu/resources/usod-dataset)



## Getting Started with Inference
Follow these steps to generate saliency maps by yourself and evaluate the result:

### Step 1: Clone the Repository
First, clone the HDANet repository using Git:
```bash
git clone https://github.com/mikurules/USOD-HDANet.git
cd USOD-HDANet
```

### Step 2: Prepare Datasets
1. Download the datasets from either Baidu Netdisk or Google Drive:
   - USOD10K dataset:  Baidu Netdisk: [USOD10K](https://pan.baidu.com/s/1edg2B9HjnHdEpmwnUOT0-w) **fetch code**: [good]  &&&  Google drive: [USOD10K](https://drive.google.com/file/d/1PH0PwKchXnkWwtAwbhNSW4utMCp5zer8/view?usp=sharing).
   
     USOD dataset:[USOD](https://irvlab.cs.umn.edu/resources/usod-dataset)
   

Place the test part of downloaded datasets in the datasets folders:
```bash
# For USOD10K
mkdir -p datasets/USOD10K && unzip usod10k.zip -d datasets/USOD10K
# For USOD
mkdir -p datasets/USOD && unzip usod.zip -d datasets/USOD
```

### Step 3: Download Pre-trained Weights
Download the pre-trained HDANet model weights from Baidu Netdisk:
-  [BaiduNetdisk](https://pan.baidu.com/s/1nndmH18X_3c_PjJXLgSysw?pwd=USOD) **fetch code**: [USOD]  &&& [Googledriven](https://drive.google.com/drive/folders/1kDGGTYmwsDH3RCFBUFBJZDYUxpPpu4G4?usp=drive_link) 

Place the downloaded checkpoint file in the `checkpoints` folder:
```bash
mkdir -p checkpoints/HDANet && unzip hda_net_weights.zip -d checkpoints/HDANet
```

### Step 4: Run Inference and Evaluation
1. Generate saliency maps using the test script:
   ```bash
   python test_produce_maps.py
   ```

2. Evaluate the model performance using the evaluation script:
   ```bash
   python test_evaluation_maps.py
   ```

After running these commands, you will find the predicted saliency maps in the `results` folder and the evaluation metrics in the console or a log file.

### Expected Output
- **Predicted Saliency Maps**: Generated maps will be saved in the `results` directory.
- **Evaluation Metrics**: Standard metrics such as MAE, F-measure, etc., will be displayed to evaluate the performance of HDANet on your dataset.

## More code
More code is currently being organized and will be released here soon.

