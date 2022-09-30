# from PIL import Image
# import torchvision as tv
# grayscale = tv.transforms.Grayscale(num_output_channels=1)

# image = Image.open('dataset/train/trainimg/00001.jpg')
# imager = Image.open('dataset/train/trainimgr/00001.jpg')

# # print(image.size) # Output: (1920, 1280)
# # #cropped_image = image.resize((640, 512))
# # box = (100, 100, 740, 612)

# # cropped_image = image.crop(box)

# # print(cropped_image.size) # Output: (1920, 1280)
# grayscale(image).save('grayscale.jpg')
# imager.save('rr.jpg')


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