# -*- coding: utf-8 -*-
from ast import expr_context
from turtle import shape
from matplotlib import testing
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
import random 
from tqdm import tqdm
from torchvision import transforms
from scipy.fftpack import hilbert as ht
import pywt
import kornia as K

grayscale = tv.transforms.Grayscale(num_output_channels=1)

def segmented_classes_extract(split, i, img_size, box):
    classes_temp = []
    for idx in range(6):
        a =  i.replace(split+"img",split+"maskscol")
        try:
            img_segm = Image.open(a.replace('.jpg', '_'+str(idx+1)+'.jpg')).convert('L')
        except:
            continue
        img_segm = img_segm.crop(box)

        # convert image to numpy array
        img_segm = np.asarray(img_segm)
        img_segm = cv2.resize(img_segm, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        test = label_preprocess(img_segm)
        c = np.array(idx+1) #class
        classes_temp.append((test,c))
    return classes_temp
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
def raw_preprocess(data, get_s=False, path_pair=None, img_size=256, ):
    box = (100, 100, 740, 612)
    if get_s:
        img_pair = Image.open(path_pair)
        img_pair = img_pair.crop(box)
        # convert image to numpy array
        img_pair = np.asarray(img_pair)
        #lasciamo qui sotto?????
        img_pair = cv2.resize(img_pair, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        data = data.astype(dtype=float)
        img_pair = img_pair.astype(dtype=float)
        out = data.copy()
        out_pair = img_pair.copy()
        
        #normalizzazione
        out = (out - out.min()) / (out.max() - out.min())
        out_pair = (out_pair - out_pair.min()) / (out_pair.max() - out_pair.min())

        return out, out_pair
    data = data.astype(dtype=float)
    # data[data<50] = 0
    out = data.copy()
    #normalizzazione?
    out = (out - out.min()) / (out.max() - out.min())

    return out

class DroneVeichleDataset(Dataset):
    def __init__(self,lp, path="dataset", 
                       split='train', 
                       modals=('img','imgr'),
                       transforms=None,
                       img_size=128,
                       to_be_loaded=False,
                       colored_data=True,
                       paired_image=False,
                       lab = False,
                       classes=False
                       ):
        super(DroneVeichleDataset, self).__init__()
        
        if not to_be_loaded:
            self.paired_image = paired_image
            self.lab = lab
            self.classes =classes
            box = (100, 100, 740, 612)
            self.img_size = img_size
            fold = split + "/"
            path1 = os.path.join(path, fold+ split+modals[0])
            path2 = os.path.join(path, fold + split+modals[1])
            list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted([os.path.join(path2, x) for x in os.listdir(path2)])
            raw_path = [] #contains RGB image real
            # print(len(list_path), list_path[200])
            for x in lp:

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
            self.raw_dataset, self.raw_classes = [], []
            self.seg_mask_dataset = []
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
                    img, img_pair = raw_preprocess(img, get_s=True, path_pair=i.replace(split+"imgr",split+"img"), img_size = self.img_size)

                    self.raw_dataset.append([(img, img_pair), c])
                    if classes:
                        self.raw_classes.append(segmented_classes_extract(split, i, self.img_size, box))

                    #?????
                    a =  i.replace(split+"imgr",split+"masksr")
                    img_segm = Image.open(a)
                    img_segm = img_segm.crop(box)

                    # convert image to numpy array
                    img_segm = np.asarray(img_segm)
                    img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                    self.seg_mask_dataset.append(label_preprocess(img_segm))
                elif c==1:
                        img = Image.open(i)
                        img = img.crop(box)

                        # convert image to numpy array
                        img = np.asarray(img)
                        # img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                        #lasciamo qui sotto?????
                        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                        img, img_pair = raw_preprocess(img, get_s=True, path_pair=i.replace(split+"img",split+"imgr"), img_size = self.img_size)

                        self.raw_dataset.append([(img, img_pair), c])
                        if classes:
                            self.raw_classes.append(segmented_classes_extract(split, i, self.img_size, box))

                        #?????
                        a =  i.replace(split+"img",split+"masks")
                        img_segm = Image.open(a)
                        img_segm = img_segm.crop(box)

                        # convert image to numpy array
                        img_segm = np.asarray(img_segm)
                        img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                        self.seg_mask_dataset.append(label_preprocess(img_segm))


            self.split = split
            self.colored_data = colored_data
            assert len(self.raw_dataset) == len(self.seg_mask_dataset)
            if classes:
                assert len(self.raw_dataset) == len(self.raw_classes)

            print("DroneVeichle "+ split+ " data load success!")
            print("total size:{}".format(len(self.raw_dataset)))
            
    def __getitem__(self, item):
        img, paired_img, class_label, seg_mask = self.raw_dataset[item][0][0],\
                                                 self.raw_dataset[item][0][1], \
                                                 self.raw_dataset[item][1], \
                                                 self.seg_mask_dataset[item]
        if self.classes:
            seg_classes= self.raw_classes[item]
        else:
            t_imgs_classes = None
        if img.shape[0]!=self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            seg_mask = cv2.resize(seg_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            paired_img = cv2.resize(paired_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            if self.classes:
                seg_classes = [cv2.resize(class_seg[0], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST) for class_seg in seg_classes]
        #trhee channels for seg mask
        # print(seg_mask.shape, img.shape)
        if len(img.shape)>2:
            seg_mask_3 = np.repeat(seg_mask[...,None],3,axis=2)
            t_img = img * seg_mask_3
            if self.classes:
                t_imgs_classes = [img * np.repeat(class_seg[0][...,None],3,axis=2)  for class_seg in seg_classes]
        else:
            t_img = img * seg_mask
            if self.classes:
                t_imgs_classes = [img * class_seg[0]  for class_seg in seg_classes]
            # print(';infrared', class_label)
        if self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                seg_mask = cv2.flip(seg_mask, 1)
                paired_img = cv2.flip(paired_img, 1)
                t_img = cv2.flip(t_img, 1)
                if self.classes:
                    seg_classes = [cv2.flip(class_seg[0],1) for class_seg in seg_classes]
                    t_imgs_classes = [cv2.flip(t_imgs_class,1) for t_imgs_class in t_imgs_classes]
        if self.lab:
            img = img/255
            t_img = t_img/255
            paired_img = paired_img/255
            if self.classes:
                t_imgs_classes = [t_imgs_class/255 for t_imgs_class in t_imgs_classes]
        else:
            # scale to [-1,1]
            img = (img - 0.5) / 0.5
            t_img = (t_img - 0.5) / 0.5
            paired_img = (paired_img - 0.5) / 0.5
            if self.classes:
                t_imgs_classes = [(t_imgs_class- 0.5) / 0.5 for t_imgs_class in t_imgs_classes]
        if len(img.shape)>2:
            img, t_img, paired_img, seg_mask, class_label, t_imgs_classes = self.get_item_rgb(img, t_img, paired_img, seg_mask, class_label,t_imgs_classes if t_imgs_classes is not None else None)
            if self.lab:
                img = K.color.rgb_to_lab(img)
                t_img = K.color.rgb_to_lab(t_img)
                paired_img = K.color.rgb_to_lab(paired_img)
                if self.classes:
                    t_imgs_classes = [K.color.rgb_to_lab(t_imgs_class) for t_imgs_class in t_imgs_classes]
                
        else:
            img, t_img, paired_img, seg_mask, class_label, t_imgs_classes = self.get_item_grey(img, t_img, paired_img, seg_mask, class_label, t_imgs_classes if t_imgs_classes is not None else None)
            if self.lab:
                img = K.color.rgb_to_lab(img)
                t_img = K.color.rgb_to_lab(t_img)
                paired_img = K.color.rgb_to_lab(paired_img)
                if self.classes:
                    t_imgs_classes = [K.color.rgb_to_lab(t_imgs_class) for t_imgs_class in t_imgs_classes]
        if self.classes:
            return img, t_img, paired_img, seg_mask, class_label, t_imgs_classes, [torch.from_numpy(lab[1]) for lab in seg_classes]

        return img, t_img, paired_img, seg_mask, class_label
        
    def __len__(self):
        return len(self.raw_dataset)

    def load_dataset(self, path="dataset", split='train', idx = "0", img_size=128, colored_data = True, paired_image=False, lab=False, classes=False):
        self.label_dataset = torch.load(f"{path}/{idx}_{split}_label_dataset.pt")
        self.raw_dataset = torch.load(f"{path}/{idx}_{split}_raw_dataset.pt")
        self.img_size = img_size
        self.split=split
        self.colored_data = colored_data
        self.paired_image = paired_image
        self.lab = lab
        self.classes = classes
    def get_item_rgb(self, img, t_img, paired_img, seg_mask, class_label,t_imgs_classes):
        if not self.colored_data:
            img = grayscale(torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1))
            t_img = grayscale(torch.from_numpy(t_img).type(torch.FloatTensor).permute(2, 0, 1))
            paired_img = grayscale(torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1))
            if t_imgs_classes is not None:
                t_imgs_classes = [grayscale(torch.from_numpy(t_img_class).type(torch.FloatTensor).permute(2, 0, 1)) for t_img_class in t_imgs_classes]

        else:
            img, t_img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1), \
                torch.from_numpy(t_img).type(torch.FloatTensor).permute(2, 0, 1)
            paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1) if len(paired_img.shape) ==3 else \
                            torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            if t_imgs_classes is not None:
                t_imgs_classes = [torch.from_numpy(t_img_class).type(torch.FloatTensor).permute(2, 0, 1) for t_img_class in t_imgs_classes]

        return img, \
            t_img , \
            paired_img, \
            torch.from_numpy(seg_mask).type(torch.LongTensor).unsqueeze(dim=0), \
            torch.from_numpy(class_label).type(torch.FloatTensor), \
            t_imgs_classes

    def get_item_grey(self, img, t_img, paired_img, seg_mask, class_label, t_imgs_classes):
        if not self.colored_data:
            img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0)
            t_img = torch.from_numpy(t_img).type(torch.FloatTensor).unsqueeze(dim=0)
            paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0)
            if t_imgs_classes is not None:
                t_imgs_classes = [torch.from_numpy(t_img_class).type(torch.FloatTensor).unsqueeze(dim=0) for t_img_class in t_imgs_classes]

        else:
            img =  torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            t_img = torch.from_numpy(t_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            if self.paired_image:
                paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1) if len(paired_img.shape) ==3 else \
                            torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
                # paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1)
            else:
                paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            if t_imgs_classes is not None:
                t_imgs_classes = [torch.from_numpy(t_img_class).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1) for t_img_class in t_imgs_classes]

        return img, \
            t_img, \
            paired_img, \
            torch.from_numpy(seg_mask).type(torch.LongTensor).unsqueeze(dim=0), \
            torch.from_numpy(class_label).type(torch.FloatTensor), \
            t_imgs_classes



# testing_dataset()


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
        random.shuffle(list_path)
        for x in list_path[:20000]:

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





def save_tensors_dataset(path="dataset", split="train", slices=19, max_length_slices=2000, img_size=256, my_folder = "dataset/tensors/tensors_paired"):
    modals=('img','imgr')
    fold = split + "/"
    path1 = os.path.join(path, fold+ split+modals[0])
    path2 = os.path.join(path, fold + split+modals[1])
    list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted([os.path.join(path2, x) for x in os.listdir(path2)])
    print(len(list_path))
    for idx in tqdm(range(slices)):
        random.shuffle(list_path)
        l_o = list_path[:max_length_slices]
        dt = DroneVeichleDataset(l_o,split=split, img_size=img_size, classes=True)
        idx = str(idx)

        torch.save(dt.raw_dataset, f"{my_folder}/{idx}_{split}_raw_dataset.pt")
        torch.save(dt.seg_mask_dataset, f"{my_folder}/{idx}_{split}_seg_mask_dataset.pt")
        torch.save(dt.raw_classes, f"{my_folder}/{idx}_{split}_raw_classes_dataset.pt")

        list_path = list(set(list_path)-set(l_o))
        print(len(list_path), " remaining samples")


# save_tensors_dataset(path="dataset", split="train", slices=19, max_length_slices=2000, img_size=256,  my_folder = "dataset/tensors/tensors_classes")
# save_tensors_dataset(path="dataset", split="val", slices=2, max_length_slices=2000, img_size=256,  my_folder = "dataset/tensors/tensors_classes")

class DefaultDataset(Dataset):
    def __init__(self, root, img_size=256, transform=None, kaist=False):
        if not kaist:
            self.samples = os.listdir(root)
            self.samples.sort()
            if "ground_truth" in self.samples:
                self.samples.remove("ground_truth")
            self.root = root
        else:
            imageset_txt = "dataset/kaist-cvpr15/imageSets/test-all-20.txt" if "val" in root else "dataset/kaist-cvpr15/imageSets/train-all-04.txt"
            with open(imageset_txt) as f:
                llp = [line.strip() for line in f.readlines()]
            path1,path2 = [],[]
            for sett in llp:
                splitted = sett.split("/")
                splitted.insert(-1,"lwir")
                final = "/".join(splitted) +".jpg"
                path1.append(final)
                path2.append(final.replace("lwir","visible"))
            self.samples = (path1+path2)
            if "ground_truth" in self.samples:
                self.samples.remove("ground_truth")
            self.root = "dataset/kaist-cvpr15/images/"
        self.transform = transform
        self.targets = None
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.box = (100, 100, 740, 612)

        self.transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(os.path.join(self.root,fname)).convert('RGB')
        img = img.crop(self.box)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)



'''
img should be a numpy array
'''


def wavelet_wrapper(wavelet_type, img, img_size, modality=None):
    if wavelet_type == 'real':
        return wavelet_real(img,img_size)
    elif wavelet_type == 'quat':
        return wavelet_quat(img,img_size, modality)
    else:
        raise Exception



@torch.no_grad()
def wavelet_quat(image,image_size, modality):
    
    ########## IMAGE ###############
    #image = imread(image)
    # image = image/255.0
    image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    # image = cv2.resize(image, (256, 256))
    image = cv2.resize(image, (image_size*2, image_size*2))

    # print sizes
    # print("Image size:", image.shape)

    gl, gh, fl, fh = get_filters()

    if args.is_best_4:
        ele, all_ = qwt(image, gl, gh, fl, fh, only_low=False, quad="all")
        subbands = []
        for subband in all_:
            subband = subband[2:,:]
            subband = np.expand_dims(subband, axis=0)
            subbands.append(subband)
        train = np.concatenate(subbands, axis=0)
        #train = torch.from_numpy(train.astype(np.float32))
        a = []
        lst_to_iterate = args.best_4
        # if modality=="ct":
        #     lst_to_iterate = args.ct_best_4
        # if modality=="t1":
        #     lst_to_iterate = args.t1_best_4
        # elif modality =="t2":
        #     lst_to_iterate = args.t2_best_4
        # else:
        #     lst_to_iterate = args.best_4
        for wav_num in lst_to_iterate:
            a.append(train[wav_num])
        q0,q1,q2,q3 = quat_mag_and_phase(*a)
        train = np.stack([q0,q1,q2,q3], axis = 0)
        return train
    else:
        q0, q1, q2, q3 = qwt(
                        image,
                        gl, gh, fl, fh, 
                        only_low=args.wavelet_quat_type == "low", 
                        quad="all"
                    )
    #q0, q1, q2, q3 = quat_mag_and_phase(q0, q1, q2, q3)

    q0, q1, q2, q3 = q0[2:,:], q1[2:,:], q2[2:,:], q3[2:,:]

    q0 = q0.reshape(q0.shape[0], q0.shape[1], 1)
    q1 = q1.reshape(q1.shape[0], q1.shape[1], 1)
    q2 = q2.reshape(q2.shape[0], q2.shape[1], 1)
    q3 = q3.reshape(q3.shape[0], q3.shape[1], 1)

    image = np.concatenate((q0, q1, q2, q3), axis=2)

    ########## MASK ###############                             
    # mask = imread(mask, as_gray=True)
    # mask = cv2.resize(mask, (256, 256))
    # mask = (mask>0.5).astype('float32')

    # mask = np.expand_dims(mask, axis=0)
    image = image.transpose(2,0,1)
    
    #print("IMAGE-->", np.shape(image))
    #print("MASK-->", np.shape(mask))
    
    #image_tensor = torch.from_numpy(image.astype(np.float32))
    #mask_tensor = torch.from_numpy(mask.astype(np.float32))

    # return tensors
    return image#, mask_tensor


@torch.no_grad()
def wavelet_real(img, image_size):
    if img.shape[1] >1:
        img = grayscale(img).squeeze().numpy()
    img = cv2.resize(img, (image_size * 2 - 4, image_size * 2 - 4))
    ll, lh, hl, hh = wavelet_transformation(img)

    qs = np.stack((ll, lh, hl, hh), axis=2)

    amp = np.linalg.norm(qs, axis=2)

    φ_num = (ll * hl + lh * hh) * 2
    φ_den = (ll * ll + lh * lh - hl * hl - hh * hh)
    φ = np.arctan(φ_num / φ_den)
    φ = np.nan_to_num(φ)

    θ_num = (ll * lh + hl * hh)
    θ_den = (ll * ll - lh * lh + hl * hl - hh * hh)
    θ = np.arctan(θ_num / θ_den)
    θ = np.nan_to_num(θ)

    ψ = 0.5 * np.arctan(2 * (ll * hh - hh * lh))
    ψ = np.nan_to_num(ψ)

    # θ = (θ - np.min(θ)) / np.ptp(θ)
    # ψ = (ψ - np.min(ψ)) / np.ptp(ψ)
    # φ = (φ - np.min(φ)) / np.ptp(φ)

    ei = np.exp(φ)
    ej = np.exp(θ)
    ek = np.exp(ψ)

    # ei = (ei - np.min(ei))/np.ptp(ei)
    # ej = (ej - np.min(ej))/np.ptp(ej)
    # ek = (ek - np.min(ek))/np.ptp(ek)
    # amp = (amp - np.min(amp))/np.ptp(amp)

    train = np.stack((amp, ei, ej, ek), axis=2)

    train = (train - np.min(train)) / np.ptp(train)

    '''
    train = qs
    train =(train - np.min(train))/np.ptp(train)
    '''
    # train = np.reshape(train, (
    # train.shape[2], train.shape[0], train.shape[1]))  # array of float32 (4,256,256) valori [0,1]
    train = np.transpose(train, (2, 0, 1))

    return train



def wavelet_transformation(img):
    # Wavelet transform of image
    # titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']

    coeffs2 = pywt.dwt2(img, 'bior1.3')  # Biorthogonal wavelet
    LL, (LH, HL, HH) = coeffs2
    # fig = plt.figure(figsize=(12, 3))
    # for i, a in enumerate([LL, LH, HL, HH]):
    #         ax = fig.add_subplot(1, 4, i + 1)
    #         ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #         ax.set_title(titles[i], fontsize=10)
    #         ax.set_xticks([])
    #         ax.set_yticks([])

    # fig.tight_layout()
    # plt.show()

    return LL, LH, HL, HH


def pywt_coeffs():
    # coefficients from pywt db8 (http://wavelets.pybytes.com/wavelet/db8/)
    gl = [-0.00011747678400228192,
    0.0006754494059985568,
    -0.0003917403729959771,
    -0.00487035299301066,
    0.008746094047015655,
    0.013981027917015516,
    -0.04408825393106472,
    -0.01736930100202211,
    0.128747426620186,
    0.00047248457399797254,
    -0.2840155429624281,
    -0.015829105256023893,
    0.5853546836548691,
    0.6756307362980128,
    0.3128715909144659,
    0.05441584224308161
    ]

    gh = [-0.05441584224308161,
    0.3128715909144659,
    -0.6756307362980128,
    0.5853546836548691,
    0.015829105256023893,
    -0.2840155429624281,
    -0.00047248457399797254,
    0.128747426620186,
    0.01736930100202211,
    -0.04408825393106472,
    -0.013981027917015516,
    0.008746094047015655,
    0.00487035299301066,
    -0.0003917403729959771,
    -0.0006754494059985568,
    -0.00011747678400228192
    ]
    return np.asarray(gl), np.asarray(gh)

# Compute Hilbert transform of the filters G_L and G_H
def get_hilbert_filters(gl, gh):
    fl = ht(gl)
    fh = ht(gh)
    return fl, fh

def get_filters():
    gl, gh = pywt_coeffs()
    fl, fh = get_hilbert_filters(gl, gh)
    return gl, gh, fl, fh



def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx* and
    *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers + 0.5), the
    ramps will have repeated max and min samples.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def as_column_vector( v):
    """Return *v* as a column vector with shape (N,1).
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    # (Shamelessly cribbed from scipy.)
    newsize = np.asanyarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _column_convolve( X, h):
    """Convolve the columns of *X* with *h* returning only the 'valid' section,
    i.e. those values unaffected by zero padding. Irrespective of the ftype of
    *h*, the output will have the dtype of *X* appropriately expanded to a
    floating point type if necessary.
    We assume that h is small and so direct convolution is the most efficient.
    """
    Xshape = np.asanyarray(X.shape)
    h = h.flatten().astype(X.dtype)
    h_size = h.shape[0]

    #     full_size = X.shape[0] + h_size - 1
    #     print("full size:", full_size)
    #     Xshape[0] = full_size

    out = np.zeros(Xshape, dtype=X.dtype)
    for idx in range(h_size):
        out += X * h[idx]
    
    outShape = Xshape.copy()
    outShape[0] = abs(X.shape[0] - h_size) + 1

    return _centered(out, outShape)



def colfilter(X, h):
    """Filter the columns of image *X* using filter vector *h*, without decimation.
    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.  If len(h) is even, each output sample is
    aligned with the mid point of each pair of input samples, and Y.shape =
    X.shape + [1 0].
    :param X: an image whose columns are to be filtered
    :param h: the filter coefficients.
    :returns Y: the filtered image.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    # Interpret all inputs as arrays
#     X = asfarray(X)
    X = np.asarray(X)
    h = as_column_vector(h)

    r, c = X.shape
    m = h.shape[0]
    m2 = np.fix(m*0.5)

    # Symmetrically extend with repeat of end samples.
    # Use 'reflect' so r < m2 works OK.
    xe = reflect(np.arange(-m2, r+m2, dtype=np.int), -0.5, r-0.5)

    # Perform filtering on the columns of the extended matrix X(xe,:), keeping
    # only the 'valid' output samples, so Y is the same size as X if m is odd.
    Y = _column_convolve(X[xe,:], h)

    return Y

def qwt(image, gl, gh, fl, fh, only_low=True, quad=1):
    '''Compute the QWT. Just compute the low frequency coefficients.
    Return L_G L_G, L_F L_G, L_G L_F, L_F L_F.'''

    if only_low:
        t1 = colfilter(image, gl)
        t1 = downsample(t1)
        lglg = colfilter(t1, gl)
        t2 = colfilter(image, fl)
        t2 = downsample(t2)
        lflg = colfilter(t2, gl)
        t3 = colfilter(image, gl)
        t3 = downsample(t3)
        lglf = colfilter(t3, fl)
        t4 = colfilter(image, fl)
        t4 = downsample(t4)
        lflf = colfilter(t4, fl)
        # lglg, lflg, lglf, lflf = t1, t2, t3, t4
        # lglg, lflg, lglf, lflf = full_quat_downsample(lglg, lflg, lglf, lflf)

        return lglg, lflg, lglf, lflf
    else:
        if quad==1:
            t1 = colfilter(image, gl)
            lglg = colfilter(t1, gl)
            t2 = colfilter(image, gl)
            lghg = colfilter(t2, gh)
            t3 = colfilter(image, gh)
            hglg = colfilter(t3, gl)
            t4 = colfilter(image, gh)
            hghg = colfilter(t4, gh)

            return lglg, lghg, hglg, hghg

        elif quad==2:
            t1 = colfilter(image, fl)
            lflg = colfilter(t1, gl)
            t2 = colfilter(image, fl)
            lfhg = colfilter(t2, gh)
            t3 = colfilter(image, fh)
            hflg = colfilter(t3, gl)
            t4 = colfilter(image, fh)
            hfhg = colfilter(t4, gh)

            return lflg, lfhg, hflg, hfhg

        elif quad==3:
            t1 = colfilter(image, gl)
            lglf = colfilter(t1, fl)
            t2 = colfilter(image, gl)
            lghf = colfilter(t2, fh)
            t3 = colfilter(image, gh)
            hglf = colfilter(t3, fl)
            t4 = colfilter(image, gh)
            hghf = colfilter(t4, fh)

            return lglf, lghf, hglf, hghf
        
        elif quad==4:
            t1 = colfilter(image, fl)
            lflf = colfilter(t1, fl)
            t2 = colfilter(image, fl)
            lfhf = colfilter(t2, fh)
            t3 = colfilter(image, fh)
            hflf = colfilter(t3, fl)
            t4 = colfilter(image, fh)
            hfhf = colfilter(t4, fh)

            return lflf, lfhf, hflf, hfhf
        
        elif quad=="all":
            t1 = colfilter(image, gl)
            t1 = downsample(t1)
            lglg = colfilter(t1, gl)
            t2 = colfilter(image, gl)
            t2 = downsample(t2)
            lghg = colfilter(t2, gh)
            t3 = colfilter(image, gh)
            t3 = downsample(t3)
            hglg = colfilter(t3, gl)
            t4 = colfilter(image, gh)
            t4 = downsample(t4)
            hghg = colfilter(t4, gh)

            t1 = colfilter(image, fl)
            t1 = downsample(t1)
            lflg = colfilter(t1, gl)
            t2 = colfilter(image, fl)
            t2 = downsample(t2)
            lfhg = colfilter(t2, gh)
            t3 = colfilter(image, fh)
            t3 = downsample(t3)
            hflg = colfilter(t3, gl)
            t4 = colfilter(image, fh)
            t4 = downsample(t4)
            hfhg = colfilter(t4, gh)

            t1 = colfilter(image, gl)
            t1 = downsample(t1)
            lglf = colfilter(t1, fl)
            t2 = colfilter(image, gl)
            t2 = downsample(t2)
            lghf = colfilter(t2, fh)
            t3 = colfilter(image, gh)
            t3 = downsample(t3)
            hglf = colfilter(t3, fl)
            t4 = colfilter(image, gh)
            t4 = downsample(t4)
            hghf = colfilter(t4, fh)

            t1 = colfilter(image, fl)
            t1 = downsample(t1)
            lflf = colfilter(t1, fl)
            t2 = colfilter(image, fl)
            t2 = downsample(t2)
            lfhf = colfilter(t2, fh)
            t3 = colfilter(image, fh)
            t3 = downsample(t3)
            hflf = colfilter(t3, fl)
            t4 = colfilter(image, fh)
            t4 = downsample(t4)
            hfhf = colfilter(t4, fh)

            # Mean of components
            # ll = (lglg + lflg + lglf + lflf)/2
            # lh = (lghg + lfhg + lghf + lfhf)/2
            # hl = (hglg + hflg + hglf + hflf)/2
            # hh = (hghg + hfhg + hghf + hfhf)/2

            ll = (lflg + lglf + lflf)/3
            lh = (lfhg + lghf + lfhf)/3
            hl = (hflg + hglf + hflf)/3
            hh = (hfhg + hghf + hfhf)/3

            # return ll, lh, hl, hh
            return (ll, lh, hl, hh), (lglg, lghg, hglg, hghg, lflg, lfhg, hflg, hfhg, lglf, lghf, hglf, hghf, lflf, lfhf, hflf, hfhf)


def quat_mag_and_phase(q0, q1, q2, q3):
    '''Compute the magnitude and phase quaternion representation.'''
    q_mp = np.asarray([q0, q1, q2, q3])

    phi = np.arctan(2*(q0+q1*q3)/(q0**2+q1**2-q2**2-q3**2))
    theta = np.arctan((q0*q1+q2*q3)/(q0**2-q1**2+q2**2-q3**2))
    psi = 1/2*np.arctan(2*(q0*q3-q3*q1))

    phi = np.nan_to_num(phi)
    theta = np.nan_to_num(theta)
    psi = np.nan_to_num(psi)

    q0_mag = np.linalg.norm(q_mp, axis=0, ord=2)
    q1_phase = np.exp(phi)
    q2_phase = np.exp(theta)
    q3_phase = np.exp(psi)

    return q0_mag, q1_phase, q2_phase, q3_phase


def downsample(component):
    return component[::2, ::2]

def full_quat_downsample(q0, q1, q2, q3):
    q0 = downsample(q0)
    q1 = downsample(q1)
    q2 = downsample(q2)
    q3 = downsample(q3)
    return q0, q1, q2, q3



class KAISTDataset(Dataset):
    def __init__(self, path="dataset", split='train', modals=('img','imgr'),transforms=None, img_size=128, to_be_loaded=False, colored_data=True, paired_image=False):
        super(KAISTDataset, self).__init__()
        imageset_txt = "dataset/kaist-cvpr15/imageSets/test-all-20.txt" if split == "val" else  "dataset/kaist-cvpr15/imageSets/train-all-04.txt"
        if not to_be_loaded:
            self.paired_image = paired_image
            self.img_size = img_size
            root = "dataset/kaist-cvpr15/images/"
            # setss = os.listdir(root)
            # for sett in setss:
            #     vvs = os.listdir(os.path.join(root,sett))
            #     for vv in vvs:
            with open(imageset_txt) as f:
                llp = [line.strip() for line in f.readlines()]
            print(len(llp))
            path1,path2, raw_path = [],[],[]
            for sett in llp:
                splitted = sett.split("/")
                splitted.insert(-1,"lwir")
                final = "/".join(splitted) +".jpg"
                path1.append(final)
                path2.append(final.replace("lwir","visible"))
            list_path = sorted(path1+path2)
            random.shuffle(list_path)
            for x in list_path[:1000]:
                if "lwir" in x:
                    c = np.array(0) #infrared
                    # tmp =  x.replace(split+"imgr",split+"masksr")
                elif "visible" in x:
                    c = np.array(1)
                    # tmp =  x.replace(split+"img",split+"masks")
                else:
                    raise Exception('wrong path probably')
                raw_path.append([x,c])
                            
            #########
            self.raw_dataset = []
            self.seg_mask_dataset = []
            #######
            self.transfroms = transforms

            for i,c in tqdm(raw_path): 
                if c == 0: #infrared
                    img = Image.open(root+i)
                    # convert image to numpy array
                    img = np.asarray(img)
                    #lasciamo qui sotto?????
                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                    img, img_pair = self.raw_preprocess(img, get_s=True, path_pair=root+ i.replace("lwir","visible"), img_size = self.img_size)

                    self.raw_dataset.append([(img, img_pair), c])
                    #?????
                    # a =  i.replace(split+"imgr",split+"masksr")
                    # img_segm = Image.open(root+a)

                    # # convert image to numpy array
                    # img_segm = np.asarray(img_segm)
                    # img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                    # self.seg_mask_dataset.append(self.label_preprocess(img_segm))
                elif c==1:
                        img = Image.open(root+i)

                        # convert image to numpy array
                        img = np.asarray(img)
                        # img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                        #lasciamo qui sotto?????
                        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                        img, img_pair = self.raw_preprocess(img, get_s=True, path_pair=root+i.replace("visible","lwir"), img_size = self.img_size)

                        self.raw_dataset.append([(img, img_pair), c])
                        #?????
                        # a =  i.replace(split+"img",split+"masks")
                        # img_segm = Image.open(root+a)

                        # # convert image to numpy array
                        # img_segm = np.asarray(img_segm)
                        # img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                        # self.seg_mask_dataset.append(self.label_preprocess(img_segm))


            self.split = split
            self.colored_data = colored_data
            print("KAIST "+ split+ " data load success!")
            print("total size:{}".format(len(self.raw_dataset)))
            
    def __getitem__(self, item):
        img, paired_img, class_label = self.raw_dataset[item][0][0],\
                                                 self.raw_dataset[item][0][1], \
                                                 self.raw_dataset[item][1], \

        if img.shape[0]!=self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            paired_img = cv2.resize(paired_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        
        if self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                paired_img = cv2.flip(paired_img, 1)
        #  scale to [-1,1]
        img = (img - 0.5) / 0.5

        if len(img.shape)>2:
            return self.get_item_rgb(img, paired_img, class_label)
        else:
            return self.get_item_grey(img, paired_img, class_label)

        
    def __len__(self):
        return len(self.raw_dataset)

    def load_dataset(self, path="dataset", split='train', idx = "0", img_size=128, colored_data = True, paired_image=False, lab=False):
        self.label_dataset = torch.load(f"{path}/{idx}_{split}_label_dataset.pt")
        self.raw_dataset = torch.load(f"{path}/{idx}_{split}_raw_dataset.pt")
        self.img_size = img_size
        self.split=split
        self.colored_data = colored_data
        self.paired_image = paired_image
        self.lab = lab
    def get_item_rgb(self, img, paired_img, class_label):
        if not self.colored_data:
            img = grayscale(torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1))
            paired_img = grayscale(torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1))
        else:
            img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1)
            paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1) if len(paired_img.shape) ==3 else \
                            torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            img=K.color.rgb_to_lab(img)
            paired_img = K.color.rgb_to_lab(paired_img)
        return img, \
            paired_img, \
            torch.from_numpy(class_label).type(torch.FloatTensor)

    def get_item_grey(self, img, paired_img, class_label):
        if not self.colored_data:
            img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0)
            paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0)
        else:
            img =  torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            if self.paired_image:
                paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1) if len(paired_img.shape) ==3 else \
                            torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
                # paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1)
            else:
                paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)

        return img, \
            paired_img, \
            torch.from_numpy(class_label).type(torch.FloatTensor)

    def label_preprocess(self,data):
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
    def raw_preprocess(self, data, get_s=False, path_pair=None, img_size=256, ):
        if get_s:
            img_pair = Image.open(path_pair)
            # convert image to numpy array
            img_pair = np.asarray(img_pair)
            #lasciamo qui sotto?????
            img_pair = cv2.resize(img_pair, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            
            data = data.astype(dtype=float)
            img_pair = img_pair.astype(dtype=float)
            out = data.copy()
            out_pair = img_pair.copy()
            
            #normalizzazione
            out = (out - out.min()) / (out.max() - out.min())
            out_pair = (out_pair - out_pair.min()) / (out_pair.max() - out_pair.min())

            return out, out_pair
        data = data.astype(dtype=float)
        # data[data<50] = 0
        out = data.copy()
        #normalizzazione?
        out = (out - out.min()) / (out.max() - out.min())

        return out
from utils import denorm
def testing_dataset():
    dt = DroneVeichleDataset(split="val", img_size=128, classes=True)
    # dt = ChaosDataset_Syn_new(path="../QWT/TarGAN/datasets/chaos2019")
    syn_loader = DataLoader(dt, shuffle=True)
    for epoch, (x_real, t_img, paired_img, mask, label_org, classes_seg, lab_seg) in enumerate(syn_loader):
    # for epoch, (x_real, paired_img, label_org) in enumerate(syn_loader):

        plt.axis('off')
        plt.subplot(241)
        plt.imshow(denorm(x_real).squeeze().cpu().numpy().transpose(1,2,0))
        plt.title('real image')
        plt.subplot(242)
        plt.imshow(denorm(t_img).squeeze().cpu().numpy().transpose(1,2,0))
        plt.title('target image')
        plt.subplot(243)
        plt.imshow(paired_img.squeeze().cpu().numpy().transpose(1,2,0),)
        plt.title('paired image')
        plt.subplot(244)
        plt.imshow(denorm(mask).squeeze().cpu().numpy(),cmap=plt.get_cmap('gray'))
        plt.title('target mask')
        # plt.savefig('test'+str(epoch))
        plt.show()
        if epoch >0:
            break
        print(lab_seg)
        for classes, l_seg in zip(classes_seg,lab_seg):
            print(classes.shape)
            plt.imshow(denorm(classes).squeeze().cpu().numpy().transpose(1,2,0))
            plt.title('target image'+str(l_seg[0]))
            plt.show()
testing_dataset()