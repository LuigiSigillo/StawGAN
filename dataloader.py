# -*- coding: utf-8 -*-
from ast import expr_context
from turtle import shape
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision as tv
grayscale = tv.transforms.Grayscale(num_output_channels=1)

class DroneVeichleDataset(Dataset):
    def __init__(self,path="dataset", split='train', modals=('img','imgr'),transforms=None, img_size=128, to_be_loaded=False, colored_data=False):
        super(DroneVeichleDataset, self).__init__()
        
        if not to_be_loaded:
            
            box = (100, 100, 740, 612)
            self.img_size = img_size
            fold = split + "/"
            path1 = os.path.join(path, fold+ split+modals[0])
            path2 = os.path.join(path, fold + split+modals[1])
            list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted([os.path.join(path2, x) for x in os.listdir(path2)])
            raw_path = [] #contains RGB image real
            # print(len(list_path), list_path[200])
            for x in list_path:

                if split+"imgr" in x:
                    c = np.array(0) #infrared
                    # tmp =  x.replace(split+"imgr",split+"masksr")
                elif split+"img" in x:
                    c = np.array(1)
                    # tmp =  x.replace(split+"img",split+"masks")
                else:
                    raise Exception('wrong path probably')
                raw_path.append([x,c])
                
            #########
            self.raw_dataset = []
            self.label_dataset = []
            #######
            self.transfroms = transforms

            for i,c in tqdm(raw_path): 
                if c == 0: #infrared
                    img = Image.open(i)
                    img = img.crop(box)
                    # convert image to numpy array
                    img = np.asarray(img)
                    #lasciamo qui sotto?????
                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                    img, img_masked = raw_preprocess(img, True)

                    self.raw_dataset.append([(img, img_masked), c])
                    #?????
                    a =  i.replace(split+"imgr",split+"masksr")
                    img_segm = Image.open(a)
                    img_segm = img_segm.crop(box)

                    # convert image to numpy array
                    img_segm = np.asarray(img_segm)
                    img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                    self.label_dataset.append(label_preprocess(img_segm))
                elif c==1:
                        img = Image.open(i)
                        img = img.crop(box)

                        # convert image to numpy array
                        img = np.asarray(img)
                        # img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                        #lasciamo qui sotto?????
                        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                        img, img_masked = raw_preprocess(img, True)

                        self.raw_dataset.append([(img, img_masked), c])
                        #?????
                        a =  i.replace(split+"img",split+"masks")
                        img_segm = Image.open(a)
                        img_segm = img_segm.crop(box)

                        # convert image to numpy array
                        img_segm = np.asarray(img_segm)
                        img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                        self.label_dataset.append(label_preprocess(img_segm))


            self.split = split
            self.colored_data = colored_data
            assert len(self.raw_dataset) == len(self.label_dataset)
            print("DroneVeichle "+ split+ " data load success!")
            print("total size:{}".format(len(self.raw_dataset)))
            
    def __getitem__(self, item):
        img, shape_mask, class_label, seg_mask = self.raw_dataset[item][0][0],\
                                                 self.raw_dataset[item][0][1], \
                                                 self.raw_dataset[item][1], \
                                                 self.label_dataset[item]

        if img.shape[0]!=self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            seg_mask = cv2.resize(seg_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            shape_mask = cv2.resize(shape_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        #trhee channels for seg mask
        # print(seg_mask.shape, img.shape)
        if len(img.shape)>2:
            seg_mask_3 = np.repeat(seg_mask[...,None],3,axis=2)
            t_img = img * seg_mask_3
            # print('normal', seg_mask_3.shape)
        else:
            t_img = img * seg_mask
            # print(';infrared', class_label)
        if self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                seg_mask = cv2.flip(seg_mask, 1)
                shape_mask = cv2.flip(shape_mask, 1)
                t_img = cv2.flip(t_img, 1)
        #  scale to [-1,1]
        img = (img - 0.5) / 0.5
        t_img = (t_img - 0.5) / 0.5

        if len(img.shape)>2:
            if not self.colored_data:
                img = grayscale(torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1))
                t_img = grayscale(torch.from_numpy(t_img).type(torch.FloatTensor).permute(2, 0, 1))
                shape_mask = grayscale(torch.from_numpy(shape_mask).type(torch.FloatTensor).permute(2, 0, 1))
            else:
                img, t_img, shape_mask = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1), \
                    torch.from_numpy(t_img).type(torch.FloatTensor).permute(2, 0, 1), \
                    torch.from_numpy(shape_mask).type(torch.FloatTensor).permute(2, 0, 1)
            return img, \
                t_img , \
                shape_mask, \
                torch.from_numpy(seg_mask).type(torch.LongTensor).unsqueeze(dim=0), \
                torch.from_numpy(class_label).type(torch.FloatTensor)
        else:
            if not self.colored_data:
                img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0)
                t_img = torch.from_numpy(t_img).type(torch.FloatTensor).unsqueeze(dim=0)
                shape_mask = torch.from_numpy(shape_mask).type(torch.FloatTensor).unsqueeze(dim=0)
            else:
                img =  torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
                t_img = torch.from_numpy(t_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
                shape_mask = torch.from_numpy(shape_mask).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)

            return img, \
                t_img, \
                shape_mask, \
                torch.from_numpy(seg_mask).type(torch.LongTensor).unsqueeze(dim=0), \
                torch.from_numpy(class_label).type(torch.FloatTensor)
    def __len__(self):
        return len(self.raw_dataset)

    def load_dataset(self, my_folder="dataset", split='train', idx = "0", img_size=128, colored_data = False):
        self.label_dataset = torch.load(f"{my_folder}/{idx}_{split}_label_dataset.pt")
        self.raw_dataset = torch.load(f"{my_folder}/{idx}_{split}_raw_dataset.pt")
        self.img_size = img_size
        self.split=split
        self.colored_data = colored_data


def label_preprocess(data):
    # # data = data.astype(dtype=int)
    # # new_seg = np.zeros(data.shape, data.dtype)
    # # new_seg[(data > 55) & (data <= 70)] = 1
    # out = data.copy()

    # return out.astype(dtype=int)

    data = data.astype(dtype=int)
    # print(data.max())
    new_seg = np.zeros(data.shape, data.dtype)
    new_seg[data >15] = 1
    # plt.imshow(new_seg,cmap='gray')
    # plt.savefig('label_prep')
    # raise Exception
    return new_seg


# return image and optioanlly the masked image, in this case we want just to remove the white border
# so the idea is to put in black all the white pixels
def raw_preprocess(data, get_s=False):

    data = data.astype(dtype=float)
    # data[data<50] = 0
    out = data.copy()
    #normalizzazione?
    out = (out - out.min()) / (out.max() - out.min())

    if get_s:
        shape_mask = out.copy()
        # plt.axis('off')
        # plt.imshow(shape_mask)
        # plt.savefig('rawp')
        # print(shape_mask[shape_mask ==1])
        # print(shape_mask.shape)
        shape_mask[shape_mask ==1] = 0 #here
        # print(shape_mask.shape)

        #640x512
        # plt.imshow(shape_mask)
        # plt.savefig('rawp_aftermask')
        # raise Exception
        return out, shape_mask
    return out


def denorm(x):
    res = (x + 1.) / 2.
    res.clamp_(0, 1)
    return res



class DroneVeichleDatasetPreTraining(Dataset):
    def __init__(self,path="dataset", split='train', modals=('img','imgr'),transforms=None, img_size=128, to_be_loaded=False, colored_data=False):
        super(DroneVeichleDatasetPreTraining, self).__init__()
        
        box = (100, 100, 740, 612)
        self.img_size = img_size
        fold = split + "/"
        path1 = os.path.join(path, fold+ split+modals[0])
        path2 = os.path.join(path, fold + split+modals[1])
        list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted([os.path.join(path2, x) for x in os.listdir(path2)])
        raw_path = [] #contains RGB image real
        # print(len(list_path), list_path[200])
        for x in list_path:

            if split+"imgr" in x:
                c = np.array(0) #infrared
                # tmp =  x.replace(split+"imgr",split+"masksr")
            elif split+"img" in x:
                c = np.array(1)
                # tmp =  x.replace(split+"img",split+"masks")
            else:
                raise Exception('wrong path probably')
            raw_path.append([x,c])
            
        #########
        self.dataset = []
        #######
        self.transfroms = transforms

        for i,c in tqdm(raw_path): 
            img = Image.open(i)
            img = img.crop(box)

            # convert image to numpy array
            img = np.asarray(img)
            # img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            #lasciamo qui sotto?????
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img = raw_preprocess(img )

            #?????
            if c==0:
                a = i.replace(split+"imgr",split+"img")
                irr = True
            else:
                a = i.replace(split+"img",split+"imgr")
                irr = False

            img_2 = Image.open(a)
            img_2 = img_2.crop(box)

            # convert image to numpy array
            img_2 = np.asarray(img_2)
            img_2 = cv2.resize(img_2, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_2 = raw_preprocess(img_2)
            if irr:
                self.dataset.append((img, img_2, np.array(1)))
            else:
                self.dataset.append((img_2, img, np.array(1)))

            irr = False
        self.split = split
        self.colored_data = colored_data
        print("DroneVeichle "+ split+ " data load success!")
        print("total size:{}".format(len(self.dataset)))
            
    def __getitem__(self, item):
        img, img_2 = self.dataset[item][0], self.dataset[item][1]
        if img.shape[0]!=self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_2 = cv2.resize(img_2, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        #trhee channels for seg mask
        # print(seg_mask.shape, img.shape)
        
        if self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                img_2 = cv2.flip(img_2, 1)

        #  scale to [-1,1]
        img = (img - 0.5) / 0.5
        img_2 = (img_2 - 0.5) / 0.5

        lab = torch.from_numpy(self.dataset[item][2]).type(torch.FloatTensor)

        if len(img.shape)>2:
            img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1)
        else:
            img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
        if len(img_2.shape)>2:
            img_2 = torch.from_numpy(img_2).type(torch.FloatTensor).permute(2, 0, 1)
        else:    
            img_2 = torch.from_numpy(img_2).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
        
        return img, img_2, lab

    def __len__(self):
        return len(self.dataset)

    def load_dataset(self, my_folder="dataset", split='train', idx = "0", img_size=128, colored_data = False):
        self.label_dataset = torch.load(f"{my_folder}/{idx}_{split}_label_dataset.pt")
        self.raw_dataset = torch.load(f"{my_folder}/{idx}_{split}_raw_dataset.pt")
        self.img_size = img_size
        self.split=split
        self.colored_data = colored_data


# dt = DroneVeichleDataset(split="val")
# # dt = ChaosDataset_Syn_new(path="../QWT/TarGAN/datasets/chaos2019")
# syn_loader = DataLoader(dt, shuffle=True)
# for epoch, (x_real, t_img, shape_mask, mask, label_org) in enumerate(syn_loader):
#     # print((x_real.shape, t_img.shape, shape_mask.shape, mask.shape, label_org.shape))
#     img = x_real.unsqueeze(dim=0)
#     pred_t1_img = t_img.unsqueeze(dim=0)
#     pred_t2_img = shape_mask.unsqueeze(dim=0)
#     pred_t3_img = mask.unsqueeze(dim=0)
#     # plt.axis('off')
#     # plt.subplot(241)
#     # plt.imshow(denorm(img).squeeze().cpu().numpy(), )
#     # plt.title('real image')
#     # plt.subplot(242)
#     # plt.imshow(denorm(pred_t1_img).squeeze().cpu().numpy())
#     # plt.title('target image')
#     # plt.subplot(243)
#     # plt.imshow(pred_t2_img.squeeze().cpu().numpy())
#     # plt.title('real image mask')
#     # plt.subplot(244)
#     # plt.imshow(denorm(pred_t3_img).squeeze().cpu().numpy(),cmap=plt.get_cmap('gray'))
#     # plt.title('target mask')
#     # plt.savefig('test'+str(epoch))
#     # # plt.show()
#     # if epoch >1:
#     #     break


# path="dataset"
# split='train'
# modals=('img','imgr')
# fold = split + "/"
# path1 = os.path.join(path, fold+ split+modals[0])
# path2 = os.path.join(path, fold + split+modals[1])
# list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted([os.path.join(path2, x) for x in os.listdir(path2)])
# import random 
# print(len(list_path))
# from tqdm import tqdm
# for idx in tqdm(range(19)):
#     random.shuffle(list_path)
#     l_o = list_path[:2000]
#     dt = DroneVeichleDataset(l_o,split=split, img_size=256)
#     my_folder = "dataset/tensors"
#     idx = str(idx)

#     torch.save(dt.raw_dataset, f"{my_folder}/{idx}_{split}_raw_dataset.pt")
#     torch.save(dt.label_dataset, f"{my_folder}/{idx}_{split}_label_dataset.pt")
#     list_path = list(set(list_path)-set(l_o))
#     print(len(list_path), " remaining samples")
# label_d = torch.load(f"{my_folder}/0label_dataset.pt")
# raw_d = torch.load(f"{my_folder}/0raw_dataset.pt")

# dt_loaded = DroneVeichleDataset(to_be_loaded=True)
# dt_loaded.load_dataset()

# for epoch, (x_real, t_img, shape_mask, mask, label_org) in enumerate(DataLoader(dt_loaded)):
#     print((x_real.shape, t_img.shape, shape_mask.shape, mask.shape, label_org.shape))



