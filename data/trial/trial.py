
# import cv2
# import matplotlib.pyplot as plt
# image = cv2.imread('frame912.jpg', 0)
# hist = cv2.calcHist([image], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.hist(image.flatten(), 256, [0, 256])
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

image = cv2.imread('frame912.jpg', 0)
# image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta, circle=True)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

fig.tight_layout()
plt.show()
