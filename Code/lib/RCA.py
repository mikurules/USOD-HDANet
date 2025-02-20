import torch
import torch.nn as nn
from Code.lib.cbam import CBAM


class RCA(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(RCA, self).__init__()
        #print("------RCA--------")
        act_fn = nn.ReLU(inplace=True)
        
        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        
        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.rgb_d = CBAM(out_dim, 1)
        self.d_rgb = CBAM(out_dim, 1)

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim+out_dim//2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)


        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth, xx):
        
        ################################
        x_rgb = self.reduc_1(rgb)
        x_dep = self.reduc_2(depth)
        
        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)
        
        x_dep_r = self.rgb_d(x_rgb1, x_dep1)
        x_rgb_r = self.d_rgb(x_dep1, x_rgb1)


        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)
        
        
        ful_mul = torch.mul(x_rgb_r, x_dep_r)
        x_in1   = torch.reshape(x_rgb_r,[x_rgb_r.shape[0],1,x_rgb_r.shape[1],x_rgb_r.shape[2],x_rgb_r.shape[3]])
        x_in2   = torch.reshape(x_dep_r,[x_dep_r.shape[0],1,x_dep_r.shape[1],x_dep_r.shape[2],x_dep_r.shape[3]])
        x_cat   = torch.cat((x_in1, x_in2),dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul,ful_max),dim=1)
        
        out1 = self.layer_ful1(ful_out)
        out2 = self.layer_ful2(torch.cat([out1,xx],dim=1))
         
        return out2

        