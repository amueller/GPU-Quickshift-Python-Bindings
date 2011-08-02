import matplotlib.pyplot as plt
from pyquickshift import quickshift
import numpy as np
im = plt.imread("flowers2.pnm")
blub = quickshift(im.astype(np.ubyte),10,6,0)
plt.imshow(blub)
plt.show()
