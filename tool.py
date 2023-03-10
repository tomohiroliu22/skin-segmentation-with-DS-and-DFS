
import numpy as np
import matplotlib.pyplot as plt

# Calculating the Dice coefficient
def IOUDICE(out,gt,c): 
    TP = np.count_nonzero((gt == c) & (out == c))
    FP = np.count_nonzero((gt != c) & (out == c))
    FN = np.count_nonzero((gt == c) & (out != c))
    Dice = 2*TP/(2*TP+FP+FN+1e-9)

    # Returning the Dice coefficient
    return Dice


def plot(img_sub, gim_sub, out_img,c):
    cimg = np.zeros((384,500,4))
    # Assigning colors to different regions of the image based on the segmentation results
    cimg[(gim_sub==c) & (out_img==c)] = [255,255,  0, 255] # True positives
    cimg[(gim_sub!=c) & (out_img==c)] = [255,  0,255, 255] # False positives
    cimg[(gim_sub==c) & (out_img!=c)] = [  0,255,255, 100] # False negatives
    cimg[(gim_sub!=c) & (out_img!=c)] = [0  ,  0,  0, 255] # True negatives

    # Displaying the original input image
    plt.imshow(img_sub,cmap="gray")
    plt.axis('off')
    plt.show()

    # Displaying the segmented image with assigned colors
    plt.imshow(cimg.astype('uint8'))
    plt.axis('off')
    plt.show()