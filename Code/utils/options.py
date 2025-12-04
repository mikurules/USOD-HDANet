import argparse

root_path='../dataset/USOD10k/TR/'  
parser = argparse.ArgumentParser()
parser.add_argument('--epoch',       type=int,   default=60,   help='epoch number')
parser.add_argument('--lr',          type=float, default=1e-4 * 1.5,  help='learning rate')#1e-4 * 1.5
parser.add_argument('--batchsize',   type=int,   default=18,    help='training batch size')#18
parser.add_argument('--trainsize',   type=int,   default=352,   help='training dataset size') #352
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,   default=25,    help='every n epochs decay learning rate')########################
parser.add_argument('--load',        type=str,   default=None,  help='train from checkpoints') 
parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')

parser.add_argument('--rgb_label_root',      type=str, default=root_path+'RGB/',           help='the training rgb images root')
parser.add_argument('--depth_label_root',    type=str, default= root_path+'depth/',         help='the training depth images root')
parser.add_argument('--gt_label_root',       type=str, default=root_path+'GT/',            help='the training gt images root')
parser.add_argument('--boundary_label_root',       type=str, default=root_path+'Boundary/',            help='the training Boundary images root')
parser.add_argument('--light_label_root',      type=str, default=root_path+'alight/',           help='the training rgb images root')


parser.add_argument('--val_rgb_root',        type=str, default='../dataset/USOD10k/VAL/RGB/',      help='the test rgb images root')
parser.add_argument('--val_depth_root',      type=str, default='../dataset/USOD10k/VAL/depth/',    help='the test depth images root')
parser.add_argument('--val_gt_root',         type=str, default='../dataset/USOD10k/VAL/GT/',       help='the test gt images root')
parser.add_argument('--val_light_root',      type=str, default='../dataset/USOD10k/VAL/alight/',      help='the test rgb images root')


parser.add_argument('--test_path',type=str, default='../dataset/USOD10k',help='test dataset path')
parser.add_argument('--save_path',           type=str, default='./Checkpoint/HDANet/',    help='the path to save models and logs')

opt = parser.parse_args()

