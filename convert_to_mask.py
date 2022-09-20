import numpy as np
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
def poly_mask(shape, bbox,*vertices, value=0):
    """
    Create a mask array filled with 1s inside the polygon and 0s outside.
    The polygon is a list of vertices defined as a sequence of (column, line) number, where the start values (0, 0) are in the
    upper left corner. Multiple polygon lists can be passed in input to have multiple,eventually not connected, ROIs.
        column, line   # x, y
        vertices = [(x0, y0), (x1, y1), ..., (xn, yn), (x0, y0)] or [x0, y0, x1, y1, ..., xn, yn, x0, y0]
    Note: the polygon can be open, that is it doesn't have to have x0,y0 as last element.

    adapted from: https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/64876117#64876117
    :param shape:    (tuple) shape of the output array (height, width)
    :param vertices: (list of tuples of int): sequence of vertices defined as
                                            [(x0, y0), (x1, y1), ..., (xn, yn), (x0, y0)] or
                                            [x0, y0, x1, y1, ..., xn, yn, x0, y0]
                                            Multiple lists (for multiple polygons) can be passed in input
    :param value:    (float or NAN)      The masking value to use (e.g. a very small number). Default: np.nan
    :return:         (ndarray) the mask array
    """
    width, height = 840,712 #shape[::-1]
    
    #width, height = 640,512
    # create a binary image
    img = Image.new(mode='L', size=(width, height), color=0)  # mode L = 8-bit pixels, black and white
    draw = ImageDraw.Draw(img)
    # draw polygons
    for polygon in vertices:
        draw.polygon(polygon, outline=1, fill=1)
    
    for box in bbox:
        draw.rectangle(box, outline=1, fill=1)
        # replace 0 with 'value'
    mask = np.array(img).astype('float32')
    mask[np.where(mask == 0)] = value
    return mask



#masks = poly_mask(None, bbox, *polygons)

with open("/home/jary/Documents/luigi/captainwho/droneveichle/dataset/val/output_val_img.json") as js:
    diz = json.load(js)

#image = {'file_name': '11882.jpg', 'height': 712, 'width': 840, 'id': 11882}
#annotation = {'iscrowd': 0, 'segmentation': [[692, 233, 729, 228, 732, 243, 695, 247]], 'category_id': 1, 'ignore': 0, 'bbox': [], 'image_id': 11882, 'id': 286792}, 

for image in tqdm(diz['images'], total=len(diz['images'])):
    anns,bboxs = [], []
    for ann in diz['annotations']:
        if ann['image_id'] == image['id']:
            if ann['segmentation'] != []:
                # print(ann)
                anns.append(ann['segmentation'][0])
            elif ann['bbox'] != []:
                # print(ann)
                bboxs.append(ann['bbox'])
    # print("----end---")
    masks = poly_mask(None, bboxs, *anns)
    # print(masks.shape)
    # plt.axis('off')
    # plt.gray()
    # plt.imshow(masks)
    # if len(bboxs)+len(anns) >20:
        # print(len(bboxs)+len(anns))
        # plt.show()
    #plt.savefig('/home/jary/Documents/luigi/captainwho/droneveichle/dataset/train/trainmasks/'+image['file_name'], dpi=120)
    imageio.imwrite('/home/jary/Documents/luigi/captainwho/droneveichle/dataset/val/valmasks/'+image['file_name'], masks)
