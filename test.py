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
a = [1,2,3,4,5,6]

for j in range(20):
    for i in a:
        print(i)
    a = [7,8,9,10,11]
