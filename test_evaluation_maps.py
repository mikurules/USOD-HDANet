import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from Code.utils.evaluator import Eval_thread
from Code.utils.dataloader import EvalDataset
import scipy.io as scio 
# from concurrent.futures import ThreadPoolExecutor

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(cfg):
    
    root_dir = cfg.root_dir
    gt_dir   = cfg.gt_dir
    
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        print("no output_dir")
        exit()
        #output_dir = root_dir
        
        
    method_names  = cfg.methods
    dataset_names = cfg.datasets
        
    
    
    threads = []
    
    for method in method_names:
        print(method)
        
        test_res = []
        log_all=''
        
        for dataset in dataset_names:
            loader = EvalDataset(osp.join(root_dir, method, dataset), osp.join(gt_dir, dataset,'GT'))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)

            ##
            print(['Evaluating----------',dataset,'----------'])
            mae,s,max_f,max_e= thread.run()   
            Evl_res=[mae,s,max_f,max_e]

            test_res.append([mae,s,max_f,max_e])
            scio.savemat(output_dir+method+'_'+dataset+'_res.mat', {'test_res':test_res})  



            log= str(['MAE:',Evl_res[0],'----- Smeansure:',Evl_res[1],'----- max_f:',Evl_res[2],'----- max_e:',Evl_res[3]])
            log+= '\n'+'{:.4f} & {:.4f} & {:.4f} & {:.4f}'.format(Evl_res[1],max_e,Evl_res[2],Evl_res[0]).replace('0.', '.')

            print(log)
            log_all+=log

    return Evl_res,log_all
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    gt_path       = './datasets/'

    sal_path      =  './test_maps/' 

    test_datasets =  ['USOD10K','USOD']#['TE']  #
    

    parser.add_argument('--methods',  type=str,  default=["HDANet"])#
    parser.add_argument('--datasets', type=str,  default=test_datasets)
    parser.add_argument('--gt_dir',   type=str,  default=gt_path)
    parser.add_argument('--root_dir', type=str,  default=sal_path)
    parser.add_argument('--save_dir', type=str,  default="./result/")
    parser.add_argument('--cuda',     type=bool, default=True)
    cfg = parser.parse_args()
    main(cfg)
