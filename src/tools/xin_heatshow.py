import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import matplotlib.image as img
import matplotlib
matplotlib.use('TkAgg')

def getPoints(heat):
	c,h,w=heat.shape
	points=[]
	for cc in range(c):
		heat_ = heat[cc]
		max_x=0
		max_y=0
		max_value=heat_[0,0]
		for i in range(1,h):
			for j in range(1,w):
				if(heat_[i,j]>max_value):
					max_value=heat_[i,j]
					max_x=i
					max_y=j
		points.append((max_x,max_y))
	return points

def getPoint(ht):
    h,w = ht.shape
    heat_ = ht
    max_x = 0
    max_y = 0
    max_value = heat_[0, 0]
    for i in range(1, h):
        for j in range(1, w):
            if (heat_[i, j] > max_value):
                max_value = heat_[i, j]
                max_x = i
                max_y = j
    return (max_x, max_y)


def norm(ht):
    min_ = np.min(ht)
    max_ = np.max(ht)
    return (ht-min_)/(max_-min_) * 255

#hm 21,56,56
def color_heat(hm):
    plt.subplot(1, 1, 1)
    image = Image.open("")
    # image = image.copy().resize((56, 56), Image.LANCZOS)
    image = image.convert('L')
    tmp = np.array(image, dtype=np.float32)
    # print(tmp.shape) # h,w,c
    hm = hm.detach().numpy()
    # points = getPoints(hm)
    # print(points)
    # print(tmp.dtype,hm.dtype)
    for k in range(hm.shape[0]):
        hmc = cv2.resize(hm[k],(224, 224))
        # hmc = cv2.applyColorMap(hmc.astype(np.uint8), cv2.COLORMAP_HOT).astype(np.float32)
        hmc = norm(hmc)
        point = getPoint(hmc)
        print(point)
        hmc[point[0], point[1]]=-1
        hmc[hmc!=-1]=0
        hmc[hmc==-1]=255
        for i in range(-2,2):
            for j in range(-2,2):
                hmc[i+point[0],j+point[1]]=255

        # hmc = np.array([hmc,hmc,hmc]).transpose((1,2,0))
        # tmp = tmp + hm[k] * 56
        tmp = cv2.add(tmp,hmc)
        cv2.imwrite(f"./hts/{k}.jpg",hmc)
        cv2.imwrite(f"./imgs/{k}.jpg",tmp)
    plt.imshow(tmp)
    cv2.imwrite("./5e.jpg",tmp)
    plt.savefig('./5e.jpg')
    plt.show()

