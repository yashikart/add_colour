# add_colour

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage import data
import matplotlib.pyplot as plt

# ✅ Load sample images from skimage (built-in)
img1 = data.astronaut()[:128, :128] / 255.0
img2 = data.coffee()[:128, :128] / 255.0
X = np.array([img1, img2], dtype=np.float32)

# Convert RGB → Lab
lab = np.array([rgb2lab(img) for img in X])
L = lab[:,:,:,0:1] / 100.0       # Lightness channel (input)
ab = lab[:,:,:,1:3] / 128.0      # Color channels (target)

# ✅ Very small CNN
model = models.Sequential([
    layers.Input(shape=(128,128,1)),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.Conv2D(2, 3, activation='tanh', padding='same')
])
model.compile(optimizer='adam', loss='mse')

# Train (quickly, just demo)
model.fit(L, ab, epochs=30, verbose=0)

# Predict colorization
pred_ab = model.predict(L)
for i in range(len(L)):
    lab_out = np.zeros((128,128,3))
    lab_out[:,:,0] = L[i,:,:,0]*100
    lab_out[:,:,1:] = pred_ab[i]*128
    rgb_out = np.clip(lab2rgb(lab_out), 0, 1)
    plt.subplot(1,2,1); plt.imshow(L[i].squeeze(), cmap='gray'); plt.title("Grayscale")
    plt.subplot(1,2,2); plt.imshow(rgb_out); plt.title("Colorized")
    plt.show()



sir practical 's




import numpy as np
import cv2
import matplotlib.pyplot as plt
net=cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt','colorizatio
#Load Cluster centers
pts_in_hull=np.load('pts_in_hull.npy',allow_pickle=True)
#Populate cluster centers at 1x1 convolutional kernel
class8=net.getLayerId('class8_ab')
conv8=net.getLayerId('conv8_313_rh')
pts_in_hull = pts_in_hull.transpose().reshape(2,313,1,1)
net.getLayer(class8).blobs=[np.full([1,313], 2.606, np.float32)]
#read the input image
image=cv2.imread('image1.jpg')
#convert the grayscale
gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#convert to rgb
gray_image_rgb=cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
#normalize the image
normalized_image=gray_image_rgb.astype('float32')/255.0
#convert to image to lab
lab_image=cv2.cvtColor(normalized_image,cv2.COLOR_RGB2Lab)
#resize the lightness channel to network input size
resized_l_channel=cv2.resize(lab_image[:,:,0],(224,224))
resized_l_channel-=50
#predict the a and b channel
net.setInput(cv2,dnn.blobFromImage(resized_l_channel))
pred=net.forward()[0,:,:,:].transpose((1,2,0))
#resize the predicted 'ab' image to the same dimension as our input image
pred_resized=cv2.resize(pred,(image.shape[1], image.shape[0]))
#concatenate the original l channel with the predicted 'ab' channels
colorized_image=cv2.cvtColor(colorized_image, cv2.COLOR_Lab2BGR)
#clip any values that fall outside the range[0,1]
colorized_image=(255*colorized_image).astype('uint8')
#save the colorized output as a 3 channel png image
output_filename='colorized_output.png'
cv2.imwrite(output_filename, colorized_image)
print(f'Colorized image saved as {output_filename}')
#show both the original grayscale and colorized image using matplotlib
plt.figure(figsize=(14,7))
#display original grayscale image
plt.subplot(1,2,1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original grayscale image')
plt.axis('off')
#display the colorized image
plt.subplot(1,2,2)
colorized_image_rgb=cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)
plt.imshow(colorized_image_rgb)
plt.title('Colorized image')
plt.axis('off')
plt.show()
