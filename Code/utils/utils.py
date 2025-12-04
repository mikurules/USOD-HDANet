def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):

    if epoch<=4:
        decay=0.2*epoch
        print("hida_lr:",decay)
    elif epoch>=45:
        decay=0.06
    else:
        #decay = decay_rate ** (epoch // decay_epoch)
        #线性衰减
        decay= 0.9*(1- epoch/45) + 0.1

    #print("hida_lr:",decay)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def adjust_lr_kos(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    if epoch<=1:
        decay= 0.2  # 1 #0.3*epoch
    elif epoch>=40:
        decay=0.04
        #print("kos_lr:",decay)
    else:
        #decay = (decay_rate ** (epoch // decay_epoch)  )*0.5 ###########################################
        #线性衰减
        decay= (0.9*(1- epoch/40) + 0.1)*0.5

    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
