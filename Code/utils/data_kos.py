import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
#from skimage.filters import threshold_multiotsu,threshold_otsu

#several data augumentation strategies
def cv_random_flip(img, label,depth,boundary,light):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        boundary=boundary.transpose(Image.FLIP_LEFT_RIGHT)
        light=light.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth,boundary,light
def randomCrop(image, label,depth,boundary,light):

    image_width = image.size[0]
    image_height = image.size[1]

    if random.random()>0.5 or min(image_width,image_height)<400:
        return image, label,depth,boundary,light

    ################################    
    # border=30  原本是固定30   
    crop_win_width = np.random.randint(image_width*0.8 , image_width)
    crop_win_height = np.random.randint(image_height*0.8 , image_height)
    #############################

    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region),boundary.crop(random_region),light.crop(random_region)


def randomRotation(image,label,depth,boundary,light):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
        boundary=boundary.rotate(random_angle, mode)
        light=light.rotate(random_angle, mode)
    return image,label,depth,boundary,light

def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):
        randX=random.randint(0,img.shape[0]-1)  
        randY=random.randint(0,img.shape[1]-1)  
        if random.randint(0,1)==0:  
            img[randX,randY]=0  
        else:  
            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,depth_root,boundary_root,light_root ,trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        
        self.boundarys=[boundary_root + f for f in os.listdir(boundary_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        
        self.lights=[light_root + f for f in os.listdir(light_root) if f.endswith('.png') or f.endswith('.jpg')]

        #print(len(self.images),len(self.gts),len(self.depths),len(self.boundarys),len(self.lights))

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths=sorted(self.depths)
        self.boundarys=sorted(self.boundarys)
        self.lights=sorted(self.lights)

        #print(len(self.images),len(self.gts),len(self.depths),len(self.boundarys),len(self.lights))
        #len(self.images) ,len(self.gts) , len(self.depth)
        self.filter_files()
        
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

        #print(len(self.images),len(self.gts),len(self.depths),len(self.boundarys),len(self.lights))


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth=self.depth_loader(self.depths[index])
        boundary=self.binary_loader(self.boundarys[index])
        light=self.rgb_loader(self.lights[index])

        image,gt,depth,boundary,light=cv_random_flip(image,gt,depth,boundary,light)


        image,gt,depth,boundary,light=randomCrop(image, gt,depth,boundary,light)
        image,gt,depth,boundary,light=randomRotation(image, gt,depth,boundary,light)

        #image=colorEnhance(image)
        # # gt=randomGaussian(gt)
        # gt=randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth =self.depths_transform(depth)
        boundary=self.gt_transform(boundary)
        light=self.img_transform(light)

        return image, gt, depth,torch.tensor([]),boundary,light  #  #torch.from_numpy(bin).float()

    def filter_files(self):
        #print(len(self.boundarys))
        #print(len(self.images),len(self.gts),len(self.boundarys),len(self.lights))
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.boundarys) and len(self.images) == len(self.lights)
        #pinry(len(self.images),len(self.gts),len(self.boundarys),len(self.lights))
        images = []
        gts = []
        depths=[]
        boundarys=[]
        lights=[]

        for img_path, gt_path,depth_path,boundary_path,light_path in zip(self.images, self.gts, self.depths,self.boundarys,self.lights):
            assert os.path.split(img_path)[-1]==os.path.split(gt_path)[-1]==os.path.split(depth_path)[-1]==os.path.split(boundary_path)[-1].replace('_edge','')==os.path.split(light_path)[-1]
            #print(os.path.split(img_path)[-1],os.path.split(gt_path)[-1],os.path.split(depth_path)[-1],os.path.split(boundary_path)[-1],os.path.split(light_path)[-1])

            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= Image.open(depth_path)
            boundary = Image.open(boundary_path)
            light=Image.open(light_path)

            if img.size == gt.size and gt.size==depth.size and depth.size==boundary.size and img.size == light.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
                boundarys.append(boundary_path)
                lights.append(light_path)
            else:
                print("-------------------filter_files fails 。。。。。。but goon----------------")
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
                boundarys.append(boundary_path)
                lights.append(light_path)                
                #exit()
        self.images = images
        self.gts = gts
        self.depths=depths
        self.boundarys=boundarys
        self.lights=lights

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    ## 深度图是16位的，范围到65000,不可以用.convert('L')
    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            depth_np = np.array(img)
            depth_np = 255 * (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))
            depth_np = depth_np.astype(np.uint8)

            depth_image = Image.fromarray(depth_np)
            return depth_image

    # def resize(self, img, gt, depth):
    #     assert img.size == gt.size and gt.size==depth.size
    #     w, h = img.size
    #     if h < self.trainsize or w < self.trainsize:
    #         h = max(h, self.trainsize)
    #         w = max(w, self.trainsize)
    #         return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
    #     else:
    #         return img, gt, depth

    def __len__(self):
        return self.size



#dataloader for training
def get_loader(image_root, gt_root,depth_root,boundary_root,light_root, batchsize, trainsize, shuffle=True, num_workers=24, pin_memory=False):

    dataset = SalObjDataset(image_root, gt_root, depth_root,boundary_root,light_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader




#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root,depth_root, light_root,testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg')]###############
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png') or f.endswith('.jpg')]
        self.lights=[light_root + f for f in os.listdir(light_root) if f.endswith('.png')  or f.endswith('.jpg')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths=sorted(self.depths)
        self.lights=sorted(self.lights)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth=self.depth_loader(self.depths[self.index])
        depth=self.depths_transform(depth).unsqueeze(0)
        light=self.rgb_loader(self.lights[self.index])
        light = self.transform(light).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        image_for_post=self.rgb_loader(self.images[self.index])
        image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size

        return image, gt, depth,light, name, np.array(image_for_post), torch.tensor([])  # #torch.from_numpy(bin).float()

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

    ###############深度图是16位的，范围到65000,不可以用.convert('L')
    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            depth_np = np.array(img)
            depth_np = 255 * (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))
            depth_np = depth_np.astype(np.uint8)

            depth_image = Image.fromarray(depth_np)
            return depth_image
    def __len__(self):
        return self.size

