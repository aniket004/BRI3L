# start with 
#cd "C:\research_code\NIPS_code\unet_ill_data_aug\unet-master"
###


from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense     
from keras import backend as K
import numpy as np
import matplotlib.image as mpimg 

import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False



#import cv2



im_width = 128
im_height = 128
border = 5


#############################
def mIOU(gt, preds):
    ulabels = np.unique(gt)
    iou = np.zeros(len(ulabels))
    for k, u in enumerate(ulabels):
        inter = (gt == u) & (preds==u)
        union = (gt == u) | (preds==u)
        iou[k] = inter.sum()/union.sum()
    return np.round(iou.mean(), 2)

#########################
    


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

    
def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model



input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

#img = cv2.imread('./white_img_seq/1.png')
#model = create_model()
#model.load_weights('model-tgs-salt.h5')

# load the best model
model.load_weights('model-tgs-salt_SSIM_epoch_25_diff_train_val.h5')



## load any test image


## Get test image ready
#test_image = image.load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/white_img_seq/1.png', target_size=(im_width, im_height))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis=0)

#test_image = test_image.resize(im_width, im_height,1)    # Ambiguity!
# Should this instead be: test_image.reshape(img_width, img_height, 3) ??


#load natural image
img = load_img('C:/research_code/NIPS_code/unet_ill_data_aug/unet-master/natural_img/9.PNG', grayscale = True)
plt.imshow(img, cmap = 'gray')




#load natural image
img = load_img('C:/research_code/NIPS_code/unet_ill_data_aug/unet-master/natural_image/11.jpg', grayscale = True)
plt.imshow(img, cmap = 'gray')




#load shifted white image
img = load_img('C:/research_code/NIPS_code/unet_ill_data_aug/unet-master/shifted_white_edited/15.png', grayscale = True)
plt.imshow(img, cmap = 'gray')



#load cornsweet image
img = load_img('C:/research_code/NIPS_code/unet_ill_data_aug/unet-master/cornsweet_illusion/6.bmp', grayscale = True)
plt.imshow(img, cmap = 'gray')




#load medical image
img = load_img('/cfarhomes/aniket04/Research_code/ECCV_results_pic/medical/129_260_img.png', grayscale = True)
plt.imshow(img, cmap = 'gray')


#load test image
img = load_img('C:/research_code/NIPS_code/unet_ill_data_aug/unet-master/herman_grid_variant/1.png', grayscale = True)
plt.imshow(img, cmap = 'gray')


#load test image
img = load_img('C:/research_code/NIPS_code/unet_ill_data_aug/unet-master/test_images/5169.png', grayscale = True)
plt.imshow(img, cmap = 'gray')


#load test image
img = load_img('C:/research_code/NIPS_code/unet_ill_data_aug/unet-master/medical_custom_data/1.jpg', grayscale = True)
plt.imshow(img, cmap = 'gray')


## load grain image
img = load_img('C:/research_code/NIPS_code/unet_ill_data_aug/unet-master/grain_images/3.jpg')
plt.imshow(img, cmap = 'gray')



#/Research_code/ECCV_results_pic/medical

# ## load test hermann grid

# img = load_img('/cfarhomes/aniket04/Research_code/ISI_work/NIPS_code/Illusion_dataset/Localization/ill_Loc/train/img/13059.png', grayscale =True)
# plt.imshow(img, cmap = 'gray')

# #load medical image
# img = load_img('/cfarhomes/aniket04/Research_code/Medical_img/Key_slice_examples/000001_03_01_088.png', grayscale = True)
# plt.imshow(img, cmap = 'gray')

##### success test cases %%%%
#000275_09_01_030.png
#000275_08_01_029.png


# ## load radiology image form journal paper
# img = load_img('/cfarhomes/aniket04/Documents/ECCV_literature/radiology_pic/25.jpeg', grayscale = True)
# plt.imshow(img, cmap = 'gray')




# #load image from white series
# img = load_img('/scratch1/NIPS_code/unet_ill_data_aug/unet-master/test_case/white_series/50.png', grayscale = True)
# plt.imshow(img, cmap = 'gray')


## load image as geometric 
#img = load_img('/scratch0/aniket/NIPS_code/unet_ill_data_aug/unet-master/geometric/14.png', grayscale = True)
#plt.imshow(img)



# # load image as grayscale SBC seq
# img = load_img('/scratch0/aniket/NIPS_code/unet_ill_data_aug/unet-master/SBC_img_seq/30.png', grayscale = True)
# plt.imshow(img)

# load image as grayscale White seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/white_seq/8.png', grayscale=True)
#plt.imshow(img)

##  load image as grayscale Howe seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/howe_seq/53.png', grayscale=True)
#plt.imshow(img)

##  load image as grayscale stepSBC seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/step_sbc/5.bmp', grayscale=True)
#plt.imshow(img)


##  load image as grayscale stepSBC seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/up_truncate/4.bmp', grayscale=True)
#plt.imshow(img)



## load image as shifted white seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/shifted_white/3.png', grayscale=True)
#plt.imshow(img)


## load image as todorovic seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/todorovic_edited/18.png', grayscale=True)
#plt.imshow(img)


## load image as shifted white seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/shifted_white_edited/8.png', grayscale=True)
#plt.imshow(img)

## load image mixed_brightness_ill.
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/mix_brightness_ill/35.png', grayscale=True)
#plt.imshow(img)


## load image as checkerboard seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/checkerboard_edited/5.png', grayscale=True)
#plt.imshow(img)

## load image as dungeon illusion
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/dungeon.png', grayscale=True)
#plt.imshow(img)

## load image as mach band seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/mach_band_seq/40.png', grayscale=True)
#plt.imshow(img)

## load image as Cornsweet illusion seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/cornsweet_illusion/2.png', grayscale=True)
#plt.imshow(img)

## load image as enhanced SBC seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/sbc_enhanced/4.png', grayscale=True)
#plt.imshow(img)


## load image as shifted white seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/shifted_white_seq/49.png', grayscale=True)
#plt.imshow(img)

## load image as strip contrast seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/strip_contrast/5.jpg', grayscale=True)
#plt.imshow(img)


## load image as surround contrast seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/surrond_contrast/5.jpg', grayscale=True)
#plt.imshow(img)


## load image as shifted white seq
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/adelson/adelson_incremental_test_patch.png', grayscale=True)
#plt.imshow(img)


## load image as white illusion
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/white/5.png', grayscale=True)
#plt.imshow(img)

##  load image as grayscale Howe seq diff width
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/howe_seq_width_20/52.png', grayscale=True)
#plt.imshow(img)


##  load image as SBC 
#img = load_img('E:/Aniket_Research/python_code/unet-grid/unet-master/SBC_seq/5.png', grayscale=True)
#plt.imshow(img)






# from numpy import expand_dims
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator
# from matplotlib import pyplot

# # load the image
# #img = load_img('bird.jpg')
# # convert to numpy array
# data = img_to_array(img)
# # expand dimension to one sample
# samples = expand_dims(data, 0)
# # create image data augmentation generator
# #datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# #datagen = ImageDataGenerator(vertical_flip = True)
# datagen = ImageDataGenerator(rotation_range = 180)
# # prepare iterator
# it = datagen.flow(samples, batch_size=1)
# # generate samples and plot
# for i in range(2):
# 	# define subplot
# 	pyplot.subplot(330 + 1 + i)
# 	# generate batch of images
# 	batch = it.next()
#     #np.squeeze(batch[0], axis=(2,))
# 	# convert to unsigned integers for viewing
# 	image = np.squeeze(batch[0], axis=(2,)).astype('uint8')
# 	# plot raw pixel data)
# 	plt.imshow(image)
# # show the figure
# plt.show()






#from keras.preprocessing.image import ImageDataGenerator
#img_gen = ImageDataGenerator(horizontal_flip= True)
#rot_img = img_gen.fit(img)
#plt.imshow(rot_img)
#img_gen.apply_transform(img)


# if colorimage, convert it to grayscale
from skimage.color import rgb2gray

# convert image to a numpy array
#img_array = np.array(image)
img_array = np.array(img)
img = rgb2gray(img_array)  # redundant if already grayscale image
img = resize(img,(128,128))
img = img[None,:,:,None]

result = model.predict(img, batch_size=1)
plt.imshow(result[0,:,:,0])


plt.imshow(result[0,:,:,0], cmap='gray', vmin=0, vmax=1)



binary_map = (result[0,:,:,0] >= 0.22) * 1    ##0.22
plt.imshow(binary_map, cmap=plt.cm.gray)


## load mask for grain image
#mask = load_img('C:/research_code/NIPS_code/unet_ill_data_aug/unet-master/grain_images/3_mask.png')
#mask = resize(rgb2gray(np.array(mask)), (128,128))
#plt.imshow(mask)
#
#th = 0.9
#miou_th = mIOU( (mask >= th), (binary_map >= th) )
#print(miou_th)



# calculate IoU between mask and predicted output



#arr = np.asarray(result)
#plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
#plt.show()

#print result
#model.predict(img)