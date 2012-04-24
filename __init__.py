import pyquickshift

def quickshift(img, tau, sigma, device=-1):
    image,segments = pyquickshift.quickshift(img, tau, sigma, device)
    return image, segments

def demo():
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    im = plt.imread(os.path.join(os.path.dirname(__file__),"lena.png"))
    image, segments = quickshift((im*255).astype(np.ubyte),10,6,-1)
    plt.subplot(131)
    plt.imshow(im)
    plt.subplot(132)
    plt.imshow(image)
    plt.subplot(133)
    plt.imshow(segments)
    plt.show()


