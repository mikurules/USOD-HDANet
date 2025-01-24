# USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection

Source code for our paper “**HDANet: Enhancing Underwater Salient Object Detection with Physics-Inspired Multimodal Joint Learning**”.

## HDANet
The HDANet addresses USOD challenges through developing targeted designs. It first integrates a task-driven underwater image enhancement module, named HydroDepthEnhanceModule (HDEM), which is based on physical models to provide enhanced images and multi modal information optimized for USOD tasks. Furthermore, we develop a physics-inspired three-way unsupervised learning strategy, leveraging the complementary effects of re-enhancement and re-degradation to improve HDEM’s generalization across diverse underwater image degradation scenarios. Additionally, we design a robust cross-attention (RCA) module to effectively fuse multimodal features while mitigating noise and blurring by exploiting channel and spatial cross-attention mechanisms. Extensive experiments on various USOD datasets demonstrate





**How to generate predicted saliency maps by yourself or retrain this model:**
You create a folder named checkpoint under the TU_USOD folder (cd TC_USOD->mkdir checkpoint) and put the [TC-USOD baseline](https://pan.baidu.com/s/1TwwaTcdmTiU2FHOC5xC3Vw) **fetch code**: [ie0k] in it to generate the predicted saliency maps (**you can also find them in the TC_USOD/preds/USOD10K in this project**). Of course, you can retrain this method with the available USOD10K dataset to get your own model.

![](fig1.png)
### Requirement
1. Python 3.8
2. Pytorch 1.6.0
3. Torchvison 0.7.0

## Benchmark
We retrained 35 SOTA methods in the fields of SOD and USOD, most of the deep methods are proposed in the years 2020, 2021, and 2022. It takes us about 1750 hours to retrain these methods. Here is the qualitative evaluation of the 35 SOTA methods and the TC-USOD baseline.


(1) **Retrained models** are available [BaiduNetdisk](https://pan.baidu.com/s/1VXyNHxy5Iy5GYYBCh_2thg) **fetch code**: [usod]  &&& [Googledriven](https://drive.google.com/file/d/1x_UhY7Ik6rFqkk4f5wNG97_CfC_DD7JZ/view?usp=drive_link) 

(2) **Predicted saliency maps of USOD10K** are available [BaiduNetdisk](https://pan.baidu.com/s/1EpnE07lgamyaUIUZWdccqA) **fetch code**: [usod] &&& [Google driven](https://drive.google.com/file/d/1D4wLLol843DEpolmO-cYpo2jaiBY7Ufn/view?usp=drive_link)

(3) **Predicted saliency maps of USOD** are available [BaiduNetdisk](https://pan.baidu.com/s/1cnmMZ0JSshssm2jc9p2BdA ) **fetch code**: [usod]  &&& [Google driven](https://drive.google.com/file/d/1YoXKUKaauy2PkkISpK-QWJpetXIsTsrO/view?usp=drive_link)



## USOD dataset

USOD10K dataset: Baidu Netdisk: [USOD10K](https://pan.baidu.com/s/1edg2B9HjnHdEpmwnUOT0-w) **fetch code**: [good]  &&&  Google drive: [USOD10K](https://drive.google.com/file/d/1PH0PwKchXnkWwtAwbhNSW4utMCp5zer8/view?usp=sharing).

USOD dataset: [USOD](https://irvlab.cs.umn.edu/resources/usod-dataset)





