import matplotlib.pyplot as plt
from pyquickshift import quickshift
import numpy as np
im = plt.imread("flowers2.pnm")
image, segments = quickshift(im.astype(np.ubyte),10,6,0)
plt.imshow(image)
plt.figure()
plt.imshow(segments)
plt.show()
