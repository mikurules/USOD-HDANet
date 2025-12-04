from Code.utils.options import opt
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
from torch import nn
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid


from Code.lib.sodnet import SODNet

from Code.utils.data_kos import get_loader,test_dataset

from Code.utils.utils import clip_gradient, adjust_lr,adjust_lr_kos
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import cv2

import argparse
###########################
from Code.lib.HDEM import MainModel as HDEM



from torch import Tensor
import random
#set the device for training
#if opt.gpu_id=='2':

#print('USE GPU 2')
from test_evaluation_maps import main as eval_main

torch.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
np.random.seed(2024)
random.seed(2024)

import shutil

cudnn.benchmark = True

#build the model
model = SODNet(32)
model_kos=HDEM()

# if(opt.load is not None):
#     model.load_state_dict(torch.load(opt.load))
#     print('load model from ',opt.load)
#     model_kos.load_state_dict(torch.load(opt.load))
#     exit()
    
model.cuda()
model_kos.cuda()
params    = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
params_kos=model_kos.parameters()
optimizer_kos = torch.optim.Adam(params_kos, opt.lr)

#set the path
train_image_root = opt.rgb_label_root
train_gt_root    = opt.gt_label_root
train_depth_root = opt.depth_label_root
train_boundary_root=opt.boundary_label_root
train_light_root = opt.light_label_root

val_image_root   = opt.val_rgb_root
val_gt_root      = opt.val_gt_root
val_depth_root   = opt.val_depth_root
val_light_root   = opt.val_light_root

save_path        = opt.save_path

ig_warning=0

guild_epoch=-2

import warnings
if ig_warning:
    warnings.filterwarnings("ignore")
if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(train_image_root, train_gt_root,train_depth_root,train_boundary_root,light_root=train_light_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader  = test_dataset(val_image_root, val_gt_root,val_depth_root, val_light_root, opt.trainsize)

#test_loader1  = test_dataset(val_image_root222, val_gt_root222,val_depth_root222,  val_light_root222, opt.trainsize)

total_step   = len(train_loader)

#set loss function
CE   = torch.nn.BCEWithLogitsLoss()

step = 0
writer     = SummaryWriter(save_path+'summary')
best_mae   = 1
best_epoch = 0

print(len(train_loader))


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()

from Code.lib import pytorch_ssim

ssim_loss_pytorch = pytorch_ssim.SSIM(window_size=7, size_average=True)
criterion = nn.BCEWithLogitsLoss()

### bce_ssim_loss
def bce_ssim_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_out = criterion(pred, mask)
    ssim_out = 1 - ssim_loss_pytorch(pred, mask)
    loss = bce_out + ssim_out
    return loss*0.5


mse_loss = nn.MSELoss()



class I_TV(torch.nn.Module):
    def __init__(self):
        super(I_TV,self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
		

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w, = x.shape
        h_tv = self.loss_fn(x[:, :, 1:, :], x[:, :, :h - 1, :])
        w_tv = self.loss_fn(x[:, :, :, 1:], x[:, :, :, :w - 1])
        loss = h_tv + w_tv
        return loss


class L_color(torch.nn.Module):
    def __init__(self):
        super(L_color, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def forward(self, x: Tensor) -> Tensor:
        x = torch.mean(x, dim=(-2, -1))
        # from color channel (0, 1, 2) corresponding to (1, 2, 0)
        loss = self.loss_fn(x, x[:, [1, 2, 0]])
        return loss

class SaturatedPixelLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        zero = a.new_zeros(1)
        one = a.new_ones(1)

        loss_max = (torch.max(a, one) + torch.max(b, one) - 2 * one).nanmean()
        loss_min = -(torch.min(a, zero) + torch.min(b, zero)).nanmean()
        loss = loss_max + loss_min
        return loss



class TransmissionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        loss = self.loss_fn(a, b)
        return loss

_color = L_color()
_img_TV = I_TV()
sp_loss=SaturatedPixelLoss()
t_loss=TransmissionLoss()

def _eval_e( y_pred, y, num):
    score = torch.zeros(num).cuda()
    thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    # else:
    #     score = torch.zeros(num)
    #     thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
    return score

def train(train_loader, epoch,save_path):#model, optimizer,
    global step
    model.train()
    model_kos.train()
    loss_all=0
    epoch_step=0
    #train_iterator=tqdm(train_loader)
    #print(len(train_loader))
    try:
        for i, (images, gts, depths, bin,boundarys,lights) in enumerate(train_loader, start=1):

            optimizer.zero_grad()
            optimizer_kos.zero_grad()
            
            images   = images.cuda()
            gts      = gts.cuda()
            depths   = depths.cuda()

            boundarys=boundarys.cuda()

            '''~~~My Framework~~~'''
            #koschmieder过程        
            image_o=images
            trans_map, atm_map______, HazefreeImage ,loss_acc= model_kos(images,not_acc=True)

            atm = torch.mean(image_o, dim=(2, 3), keepdim=True)

            atm_map = atm.expand_as(image_o)


            ########loss_re_en#########
            loss_re_en=0
            HazefreeImage_XX = HazefreeImage
            already_trans_map, already_atm_map______, already_HazefreeImage ,already_loss_acc= model_kos(HazefreeImage_XX,not_acc=True)
            # 创建一个1的张量
            allone_tensor = torch.ones_like(already_trans_map)
            # 计算MSE损失
            loss_re_en = mse_loss(already_trans_map, allone_tensor   )
            #
            if loss_re_en<0.0001:
                loss_re_en=0
                
            ##########################################

            _alaph =  random.uniform(0.85, 0.95)  #px
            px = _alaph
            InputimageX = image_o*_alaph + (1 - _alaph)*atm_map
            trans_mapX, atm_mapX, HazefreeImageX,loss_acc_none = model_kos(InputimageX,not_acc=True)

            assert loss_acc_none==None

            loss_sp=sp_loss(HazefreeImage,HazefreeImageX)
            lossT = t_loss(_alaph*trans_map,trans_mapX)


            #灰色世界假设损失
            lossCLR = _color(HazefreeImage)
                    
            #总变异损失
            lossI = _img_TV(HazefreeImage)
            
            loss_kos =   lossCLR + 500*lossT  + 0.1*loss_sp +   0.1*lossI  + 2* loss_re_en # +  4* loss__ssim #+ loss_t1mult2 #+loss__ssim #  #+  0.1*loss_acc

            loss_guild=0
            ######################
            # trans_map归一化处理
            min_val = trans_map.min()
            max_val = trans_map.max()
            normalized_trans_map = (trans_map - min_val) / (max_val - min_val)

            #pre_res  = model(HazefreeImage,trans_map, bin)
            pre_res  = model(HazefreeImage,normalized_trans_map, bin)
            #pre_res  = model(HazefreeImage,depth_trans, bin)
            #pre_res  = model(HazefreeImage,depths, bin)
            
            loss1    = structure_loss(pre_res[0], gts) 
            loss2    = structure_loss(pre_res[1], gts)
            loss3    = structure_loss(pre_res[2], gts) 
            loss1u = iou_loss(pre_res[0], gts)
            loss2u = iou_loss(pre_res[1], gts)
            loss3u = iou_loss(pre_res[2], gts)

            loss3r    = structure_loss(pre_res[3], gts) 
            loss4r    = structure_loss(pre_res[4], gts)
            loss5r    = structure_loss(pre_res[5], gts) 
            loss6r    = structure_loss(pre_res[6], gts) 
            loss3ru = iou_loss(pre_res[3], gts)
            loss4ru = iou_loss(pre_res[4], gts)
            loss5ru = iou_loss(pre_res[5], gts)
            loss6ru = iou_loss(pre_res[6], gts)

            loss3d    = structure_loss(pre_res[7], gts) 
            loss4d    = structure_loss(pre_res[8], gts)
            loss5d    = structure_loss(pre_res[9], gts) 
            loss6d    = structure_loss(pre_res[10], gts) 
            loss3du = iou_loss(pre_res[7], gts)
            loss4du = iou_loss(pre_res[8], gts)
            loss5du = iou_loss(pre_res[9], gts)
            loss6du = iou_loss(pre_res[10], gts)

            loss3m    = structure_loss(pre_res[11], gts) 
            loss4m    = structure_loss(pre_res[12], gts)
            loss5m    = structure_loss(pre_res[13], gts) 
            loss6m    = structure_loss(pre_res[14], gts) 
            loss3mu = iou_loss(pre_res[11], gts)
            loss4mu = iou_loss(pre_res[12], gts)
            loss5mu = iou_loss(pre_res[13], gts)
            loss6mu = iou_loss(pre_res[14], gts)

            ##最终预测图是pre_res[2]

            #bce_ssim_loss(pre_res[2], boundarys)

            if epoch>5:
                # loss_edge_1=bce_ssim_loss(pre_res[15], boundarys)
                # loss_edge_2=bce_ssim_loss(pre_res[16], boundarys)
                # #loss_edge_2=0
                # loss_edge_3=bce_ssim_loss(pre_res[17], boundarys)
                loss_edge_1=bce_ssim_loss(pre_res[0], boundarys)
                loss_edge_2=bce_ssim_loss(pre_res[1], boundarys)
                loss_edge_3=bce_ssim_loss(pre_res[2], boundarys)

                loss_edge=loss_edge_1+loss_edge_2+loss_edge_3
            else:
                loss_edge=0

            
            ########################
            loss_seg = loss1 + loss2 + loss3 + loss1u + loss2u + loss3u  \
                   + 0.8 * (loss3r + loss3ru + loss3d + loss3du + loss3m + loss3mu) \
                   + 0.6 * (loss4r + loss4ru + loss4d + loss4du + loss4m + loss4mu) \
                   + 0.4 * (loss5r + loss5ru + loss5d + loss5du + loss5m + loss5mu) \
                   + 0.2 * (loss6r + loss6ru + loss6d + loss6du + loss6m + loss6mu)


            if epoch<=5:
                loss = loss_seg*( epoch*0.2)  + loss_kos # +5*loss_re_en
            else:
                loss = loss_seg+loss_kos +loss_edge
            loss.backward()
            ################################

            clip_gradient(optimizer, opt.clip)
            clip_gradient(optimizer_kos, opt.clip)
            optimizer.step()
            optimizer_kos.step()


            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if epoch==1:
                show_times=30
            else:
                show_times=100
            if i % show_times == 0 or i == total_step or i==1:
                if epoch<=-2:
                    print(progress)
                print('{} Epoch [{:03d}/{:03d}] Step [{:04d}/{:04d}] Loss1:{:.4f} Loss2:{:0.4f} Loss3:{:0.4f}   Loss_kos:{:0.4f} loss_guild:{:0.4f} loss_re_en:{:0.4f} loss_edge:{:0.4f}  loss:{:0.4f}'.
                    format(datetime.now().strftime("%m-%d %H:%M:%S"), epoch, opt.epoch, i, total_step, loss1.data, loss2.data,  loss3.data , loss_kos.data if loss_kos else 0,loss_guild.data if loss_guild else 0,loss_re_en.data if loss_re_en else -1,loss_edge if loss_edge else -1,loss.data))
                logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f} Loss_kos: {:0.4f} loss_guild: {:0.4f} loss_re_en: {:0.4f} loss_edge:{:0.4f} loss:{:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data,  loss3.data , loss_kos.data if loss_kos else 0,loss_guild.data if loss_guild else 0,loss_re_en.data if loss_re_en else -1,loss_edge if loss_edge else -1 ,loss.data))
                myiter=len(train_loader)*epoch+i
                writer.add_scalar('Loss/loss1', loss1.item(), myiter)
                writer.add_scalar('Loss/loss2', loss2.item(), myiter)
                writer.add_scalar('Loss/kos', loss_kos.item() if loss_kos else 0, myiter)
                writer.add_scalar('Loss/loss3', loss3.item(), myiter)
                
        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        
        # if epoch in [30,40]:#(epoch % 5) == 0 and (epoch>=25):
        #     save_opt=1
        #     if save_opt==1:
        #         torch.save( optimizer.state_dict(), save_path + 'HyperNet_optimizer_epoch_{}.pth'.format(epoch))
        #         torch.save(optimizer_kos.state_dict(), save_path + 'kosNet_optimizer_epoch_{}.pth'.format(epoch))

        if epoch in [30,40,45,50]:#(epoch % 5) == 0 and (epoch>=25):
            torch.save(model.state_dict(), save_path+'HyperNet_epoch_{}.pth'.format(epoch))
        
            torch.save(model_kos.state_dict(), save_path+'kosNet_epoch_{}.pth'.format(epoch))
            

            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #torch.save(model.state_dict(), save_path+'HyperNet_epoch_{}.pth'.format(epoch+1))
        #torch.save( optimizer.state_dict(), save_path + 'HyperNet_optimizer_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
        
        
        
#test function
def val(test_loader, epoch,save_path):#model,
    global best_mae,best_epoch
    model.eval()
    model_kos.eval()
    with torch.no_grad():
        mae_sum=0.0
        Em = torch.zeros(255).cuda()
        for i in range(test_loader.size):
            image, gt,depth,light,  name,img_for_post, bin = test_loader.load_data()
            #gt      = np.asarray(gt, np.float32)
            gt = torch.from_numpy(np.asarray(gt, np.float32)).cuda()

            gt     /= (gt.max() + 1e-8)
            image   = image.cuda()
            depth   = depth.cuda()

            trans_map, atm_map______, HazefreeImage,loss_acc =model_kos(image,not_acc=True)
            ####################################
            # trans_map归一化处理
            min_val = trans_map.min()
            max_val = trans_map.max()

            normalized_trans_map = (trans_map - min_val) / (max_val - min_val)
            #####################################################

            pre_res = model(HazefreeImage,normalized_trans_map, bin)###############################


            res     = pre_res[2]
            res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            #res     = res.sigmoid().data.cpu().numpy().squeeze()
            #转为tensor操作
            res = torch.sigmoid(res).squeeze()
            res     = (res - res.min()) / (res.max() - res.min() + 1e-12)
            #mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])

            ######
            #mae_sum += torch.sum(torch.abs(res - gt)).item()*1.0 / (gt.shape[0] * gt.shape[1])
            mae_sum += torch.abs(res - gt).mean()

            
            if epoch>30:
                Em+=_eval_e(res, gt, 255)

            
        mae_sum1=0
        mae = mae_sum/test_loader.size #+ mae_sum1/test_loader1.size
        max_e = (Em / test_loader.size).max().item()


        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        writer.add_scalar('max_e', torch.tensor(max_e), global_step=epoch)
        print('Epoch: {} MAE: {} max_e:{} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,max_e,best_mae,best_epoch))
        if epoch==1:
            best_mae = mae
        else:
            if mae<best_mae:
                best_mae   = mae
                best_epoch = epoch
                print('best epoch:{}'.format(epoch))                
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))


if __name__ == '__main__':
    print("Start train...")
    
    for epoch in range(1, opt.epoch):
        
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate_hida', cur_lr, global_step=epoch)

        cur_lr_kos = adjust_lr_kos(optimizer_kos, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        print(cur_lr_kos)
        writer.add_scalar('learning_rate_kos', cur_lr_kos, global_step=epoch)
        
        # train
        train(train_loader,  epoch,save_path)#model, optimizer,
        #test
        if epoch%4==0:
            val(test_loader, epoch=epoch,save_path=save_path)#model,

