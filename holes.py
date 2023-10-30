import numpy as np 
import matplotlib.pyplot as plt
from skimage.morphology import (binary_erosion, binary_dilation, binary_closing, binary_opening)
from skimage.measure import label 

STARS=np.load("psnpy.txt")

def un_mask(mask)->int:
    stars_new=(binary_erosion(STARS,mask))
    labeled=label(stars_new)
    num_components = np.max(labeled) 
    return num_components 

right_mask=np.array([[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]])
down_mask=np.array([[1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1 , 1],
                    [1, 1, 0, 0, 1, 1],
                    [1, 1, 0, 0, 1, 1]])
up_mask=np.array([[1, 1, 0, 0, 1, 1],
                    [1, 1, 0, 0, 1 , 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1]])
no_mask=np.array([[1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1 , 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1]])
left_mask=np.array([[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]])

COUNT = {
    "nohole": un_mask(no_mask),
    "uphole": un_mask(up_mask)-un_mask(no_mask),
    "downhole": un_mask(down_mask)-un_mask(no_mask),
    "lefthole": un_mask(left_mask),
    "righthole": un_mask(right_mask)
}

for mask in COUNT.keys():
    print(f"{mask} {COUNT[mask]}")
print(f"sum {sum(COUNT.values())}")

plt.imshow(STARS)
plt.show()