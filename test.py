import matplotlib.pyplot as plt
from quickshift_py import quickshift
import numpy as np
im = plt.imread("flowers2.pnm")
blub = quickshift(im.astype(np.ubyte),10,6,0)
plt.imshow(blub.reshape(3,320*213).T.reshape(320,213,3))
plt.show()
