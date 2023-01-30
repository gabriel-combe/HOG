import cv2
import numpy as np
import matplotlib.pyplot as plt

BLOCK_SIZE = 8
BLOCK_NORM_SIZE = 16
NB_BIN = 9

img = cv2.imread('images/angry_cat.jpg')
img = np.transpose(img, (1, 0, 2))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = img.astype(np.float32)
image = cv2.resize(image, (488, 496), interpolation = cv2.INTER_AREA)

kernely = np.array([[-1., 0., 1.]])
kernelx = np.transpose(kernely)

image = np.sqrt(image)
Gx = cv2.filter2D(image, cv2.CV_32F, kernelx)
Gy = cv2.filter2D(image, cv2.CV_32F, kernely)

mag = np.sqrt(Gx**2 + Gy**2)
ang = np.arctan2(Gy, Gx)

maxindexes = np.argmax(mag, axis=-1)[..., np.newaxis]
magmax = np.take_along_axis(mag, maxindexes, axis=-1).squeeze(-1)
angmax = np.take_along_axis(ang, maxindexes, axis=-1).squeeze(-1)
angmax = np.rad2deg(angmax)

histo = np.zeros((int(image.shape[0]/BLOCK_SIZE), int(image.shape[1]/BLOCK_SIZE), NB_BIN))

step = 180/NB_BIN
bin0 = (angmax % 180) // step
bin0[np.where(bin0>=NB_BIN)]=0
bin1 = bin0 + 1
bin1[np.where(bin1>=NB_BIN)]=0

plt.subplot(321), plt.imshow(img), plt.title('Image')
plt.subplot(323), plt.imshow(Gx), plt.title('Gradient X')
plt.subplot(324), plt.imshow(Gy), plt.title('Gradient Y')
plt.subplot(325), plt.imshow(magmax), plt.title('Magnitude image')
plt.subplot(326), plt.imshow(angmax), plt.title('Orientation image')

plt.show()