from PIL import Image
import torchvision as tv
import kornia as K
import torch
import numpy as np
import sys
import cv2

from dataloader import label_preprocess
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt
grayscale = tv.transforms.Grayscale(num_output_channels=1)
image = Image.open('dataset/train/trainimg/00001.jpg')
imager = Image.open('dataset/train/trainimgr/00001.jpg')


color_palette = [(255,0,0), (128,0,128), (0,192,0), (0,100,0), (200,200,0)]
dict_palette = {"car":color_palette[0], 
                "feright_car": color_palette[1],
                "truck": color_palette[2],
                "bus" :color_palette[3],
                "van":color_palette[4]}

threesolds = [(240,255), (113,145), (177,207), (85,115), (185,215)]
split='val'
i = 'dataset/val/valimg/01469.jpg'
raw_classes, classes_temp=[],[]
box = (100, 100, 740, 612)
# image_maskr = Image.open('dataset/val/valmaskscolr/00021.jpg').convert("RGB")
for idx in range(6):
    a =  i.replace(split+"img",split+"maskscol")
    try:
        img_segm = Image.open(a.replace('.jpg', '_'+str(idx+1)+'.jpg')).convert('L')
    except:
        continue
    img_segm = img_segm.crop(box)

    # convert image to numpy array
    img_segm = np.asarray(img_segm)
    img_segm = cv2.resize(img_segm, (256, 256), interpolation=cv2.INTER_LINEAR)

    test = label_preprocess(img_segm)
    print(idx+1)
    c = np.array(idx+1) #class
    classes_temp.append((test,c))

raw_classes.append(classes_temp)
    # plt.imshow(test,cmap='gray')
    # plt.show()


# for i,(min_t,max_t) in enumerate(threesolds):
#     new_a = np.zeros(np_img.shape)
#     print(np.count_nonzero( (min_t<np_img) & (np_img<max_t) ) )
#     new_a[(min_t<np_img) & (np_img<max_t)] = color_palette[i][0] if color_palette[i][0]!= 0 else color_palette[i][1]
#     # print(new_a)

#     test = Image.fromarray(new_a.astype("uint8"))
#     test.save('rr'+list(dict_palette.keys())[i]+".jpg")


# blackPxNum = np.count_nonzero([np.asarray(img)<=76]) #number of black pixels
# whitePxNum = np.asarray(img).size - blackPxNum 
# print(whitePxNum)
# print(blackPxNum)
# if whitePxNum<100:
#     """segment is black"""
#     print("blac")
# im = torch.tensor(np.asarray(image))
# im = im.permute(2,0,1)
# lab_image = K.color.rgb_to_lab(im/255.0)
# lab_img_np = lab_image.numpy().transpose(1,2,0)
# print(image.size) # Output: (1920, 1280)
# #cropped_image = image.resize((640, 512))
# box = (100, 100, 740, 612)

# cropped_image = image.crop(box)
# print(cropped_image.size) # Output: (1920, 1280)
# print(lab_image.shape)
# grayscale(image).save('grayscale.jpg')
# imager.save('rr.jpg')
# img_np = np.asarray(image)

# print(img_np.shape)

# plt.subplot(251)
# plt.imshow(img_np[:,:,0]/255.0 )
# plt.subplot(252)
# plt.imshow(img_np[:,:,1]/255.0, )
# plt.subplot(253)
# plt.imshow(img_np[:,:,2]/255.0, )
# plt.subplot(254)
# plt.imshow(img_np, )
# rgbr = K.color.lab_to_rgb(lab_image)
# img_np_rgb = rgbr.numpy().transpose(1,2,0)

# print(rgbr.shape)
# plt.subplot(255)
# plt.imshow(rgbr, )


# import matplotlib.pyplot as plt

# count = 1
# NSAMPLE=2
# fig = plt.figure(figsize=(12,3*NSAMPLE))
# for rgb in [lab_img_np, img_np_rgb]:
#     ## This section plot the original rgb
#     ax = fig.add_subplot(NSAMPLE,4,count)
#     ax.imshow(rgb); ax.axis("off")
#     ax.set_title("original LAB")
#     count += 1
    
#     for i, lab in enumerate(["L","A","B"]):
#         crgb = np.zeros(rgb.shape)
#         crgb[:,:,i] = rgb[:,:,i]
#         ax = fig.add_subplot(NSAMPLE,4,count)
#         ax.imshow(crgb); ax.axis("off")
#         ax.set_title(lab)
#         count += 1
    
# plt.savefig("aa")




# import os
# import numpy as np
# root = "dataset/kaist-cvpr15/images"
# # setss = os.listdir(root)
# # for sett in setss:
# #     vvs = os.listdir(os.path.join(root,sett))
# #     for vv in vvs:
# with open("dataset/kaist-cvpr15/imageSets/train-all-04.txt") as f:
#     llp = [line.strip() for line in f.readlines()]
# print(len(llp))
# path1,path2, raw_path = [],[],[]
# for sett in llp:
#     splitted = sett.split("/")
#     splitted.insert(-1,"lwir")
#     final = "/".join(splitted) +".jpg"
#     path1.append(final)
#     path2.append(final.replace("lwir","visible"))
# list_path = sorted(path1+path2)

# print(len(list_path))
# for x in list_path:
#     if "lwir" in x:
#         c = np.array(0) #infrared
#         # tmp =  x.replace(split+"imgr",split+"masksr")
#     elif "visible" in x:
#         c = np.array(1)
#         # tmp =  x.replace(split+"img",split+"masks")
#     else:
#         raise Exception('wrong path probably')
#     raw_path.append([x,c])

# print(raw_path[2])        


