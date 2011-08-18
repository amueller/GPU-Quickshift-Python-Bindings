import pyquickshift

def quickshift(img, tau, sigma, device=-1):
    image,segments = pyquickshift.quickshift(img, tau, sigma, device)
    return image[::-1,:], segments

def demo():
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    im = plt.imread(os.path.join(os.path.dirname(__file__),"flowers2.pnm"))
    image, segments = quickshift(im.astype(np.ubyte),10,6,0)
    plt.imshow(image)
    plt.figure()
    plt.imshow(segments)
    plt.show()


