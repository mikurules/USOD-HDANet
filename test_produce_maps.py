import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import os, argparse
import cv2
######################################
from Code.lib.sodnet import SODNet
from Code.utils.data_kos import test_dataset
#######################################
import time
from Code.lib.HDEM import MainModel as HDEM
#######################################
from test_evaluation_maps import main as eval_main
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str, default='./datasets/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
#if opt.gpu_id=='0':
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#print('USE GPU 0')
 

#load the model
model = SODNet(32)
model.cuda()
model_kos=HDEM()
model_kos.cuda()

######################################
test_name= "HDANet"     
save_name="HDANet"
####################################

model.load_state_dict(torch.load('./Checkpoint/'+test_name+'/USODNet.pth'))

model_kos.load_state_dict(torch.load('./Checkpoint/'+test_name+'/HDEM.pth'))

model.eval()
model_kos.eval()


test_datasets = ['USOD10K','USOD'] #
and_eval=0  #eval after infer


for dataset in test_datasets:
    save_path = './test_maps/'+save_name+'/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    image_root  = dataset_path + dataset + '/RGB/'
    gt_root     = dataset_path + dataset + '/GT/'
    depth_root  = dataset_path + dataset + '/depth/'
    light_root=dataset_path + dataset + '/RGB/'

    test_loader = test_dataset(image_root, gt_root,depth_root,light_root, opt.testsize)
    times = []
    for i in range(test_loader.size):
        if i%100==0:
            print(i)
        image, gt,depth,light, name, image_for_post, bin = test_loader.load_data()

        gt      = np.asarray(gt, np.float32)
        gt     /= (gt.max() + 1e-16)
        image   = image.cuda()
        depth   = depth.cuda()
        bin = bin.cuda()
        s = time.time()

        trans_map, atm_map______, HazefreeImage ,loss_acc= model_kos(image,not_acc=True)

        min_val = trans_map.min()
        max_val = trans_map.max()
        normalized_trans_map = (trans_map - min_val) / (max_val - min_val)

        pre_res = model(HazefreeImage,normalized_trans_map, bin)

        end = time.time()
        times.append(end-s)
        res     = pre_res[2] 


        res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res     = res.sigmoid().data.cpu().numpy().squeeze()
        res     = (res - res.min()) / (res.max() - res.min() + 1e-16)

        res_path=save_path+name

        cv2.imwrite(res_path,res*255)


    print('Test Done!')
    time_sum = 0
    for i in times:
        time_sum += i
    print("FPS: %f" % (1.0 / (time_sum / len(times))))

# if and_eval:
#     parser2 = argparse.ArgumentParser()
#     parser2.add_argument('--methods',  type=str,  default=[save_name])#  
#     parser2.add_argument('--datasets', type=str,  default=['TE','USOD'])
#     parser2.add_argument('--gt_dir',   type=str,  default='/home/lang/LYW/dataset/USOD10k/')
#     parser2.add_argument('--root_dir', type=str,  default='/home/lang/LYW/HDSNet/test_maps/')
#     parser2.add_argument('--save_dir', type=str,  default="/home/lang/LYW/HDSNet/result/")
#     parser2.add_argument('--cuda',     type=bool, default=True)
#     cfg2 = parser2.parse_args()
#     HIDAres,eval_log=eval_main(cfg2)  #[mae,s,max_f,max_e]