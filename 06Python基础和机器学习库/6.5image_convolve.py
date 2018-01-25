import numpy as np
import os
from PIL import Image
def convolve(image,weight):
    height,width = image.shape
    h,w = weight.shape
    height_new = height - h + 1
    width_new = width - w + 1
    image_new = np.zeros((height_new,width_new))
    for i in range(height_new):
        for j in range(width_new):
            image_new[i,j]= np.sum(image[i:h+i,j:w+j]*weight)
    image_new = image_new.clip(0,255)
    image_new = np.rint(image_new).astype('uint8')
    return image_new

if __name__=="__main__":
    im = 'son.png'
    A = Image.open(im)
    outpath = '.\\SonConPic\\'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    a = np.array(A)
    # print(a.shape)
    # 卷积核
    soble_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    soble_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    soble = np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]))
    prewitt_x = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    prewitt_y = np.array(([-1, -1,-1], [0, 0, 0], [1, 1, 1]))
    prewitt = np.array(([-2, -1, 0], [-1, 0, 1], [0, 1, 2]))
    laplacian = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))
    laplacian2 = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]))
    weight_list = ('soble_x', 'soble_y', 'soble', 'prewitt_x', 'prewitt_y', 'prewitt', 'laplacian', 'laplacian2')
    for weight in weight_list:
        print(weight+'R')
        R = convolve(a[:,:,0],eval(weight))
        print('G')
        G = convolve(a[:,:,1],eval(weight))
        print('B')
        B = convolve(a[:,:,2],eval(weight))
        I = 255 - np.stack((R,G,B),2)
        Image.fromarray(I).save(outpath+weight+"_"+im)

