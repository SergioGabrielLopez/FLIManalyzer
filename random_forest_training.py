# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:37:00 2024

@author: lopez
"""

# We import the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, img_as_float, filters, feature
from scipy import ndimage as ndi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

# We import the image.
img = img_as_float(io.imread('C:/Users/lopez/Documents/Desktop old PC/AI MSC/AI Capstone Proposal/Intensity images/Pollen_1.tif', as_gray=True))

# We crop the image.
img = img[2:62,2:62]

# We create an empty DataFrame.
data = pd.DataFrame()

# We add the original image as the first feature.
orig_flat = img.reshape(-1)

# We add it to the DataFrame. 
data['Original image'] = orig_flat

# We generate Gabor filters and add them to the features.
num = 1 
kernels = []
for theta in range(4):
    theta = theta / 4.0 * np.pi
    for sigma in (1, 3):
        for frequency in np.arange(0.05, 0.25, 0.05):
            gabor_label = 'Gabor ' + str(num)
            kernel = np.real(filters.gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
            convolved = ndi.convolve(img, kernel, mode='wrap')
            convolved_flat = convolved.reshape(-1)
            data[gabor_label] = convolved_flat
            num += 1

# We add Canny filters to the features.
num = 1
for sigma in np.arange(1,9,2):
    canny_label = 'Canny ' + str(num)
    cannied = feature.canny(img, sigma=sigma)
    cannied_flat = cannied.reshape(-1)
    data[canny_label] = cannied_flat*1.
    num += 1
    
# We add a Sobel filter to the features.
sobel = filters.sobel(img)
sobel_flat = sobel.reshape(-1)
data['Sobel'] = sobel_flat

# We add a Prewitt filter to the features.
prewitt = filters.prewitt(img)
prewitt_flat = prewitt.reshape(-1)
data['Prewitt'] = prewitt_flat

# We add a Scharr filter to the features.
scharr = filters.scharr(img)
scharr_flat = scharr.reshape(-1)
data['Scharr'] = scharr_flat

# We add a Roberts filter to the features.
roberts = filters.roberts(img)
roberts_flat = roberts.reshape(-1)
data['Roberts'] = roberts_flat

# We add a few Gaussian filters to the features.
num = 1
for sigma in (1,2,3,4,5,6,7,8,9,10):
    gaussian_label = 'Gaussian ' + str(num)
    gaussian_img = filters.gaussian(img, sigma=sigma)
    gaussian_flat = gaussian_img.reshape(-1)
    data[gaussian_label] = gaussian_flat
    num += 1
    
# We import the labelled image.
label_img = img_as_float(io.imread('C:/Users/lopez/Documents/Desktop old PC/AI MSC/AI Capstone Proposal/Intensity images/Labels/Pollen_1_label.tif', as_gray=True))

# We crop the labelled image.
label_img = label_img[2:62,2:62]

# We flatten this image and add it to the DataFrame.
label_img_flat = label_img.reshape(-1)
data['Label'] = label_img_flat

# We declare the independent and dependent variables.
data_array = data.values
y = data_array[:,-1:].reshape(-1)
X = data_array[:,:-1]

# We split the data into test and train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# We declare the model.
model = RandomForestClassifier(n_estimators=100, random_state=23)
model.fit(X_train, y_train)

# We evaluate the accuracy.
y_pred = model.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print('\nThe accuracy is {:.2f}.'.format(acc))

# We obtain the feature importance.
feat_imp = list(model.feature_importances_)

# We get the list of features.
features_list = data.columns[:-1]

#We visualize the importances.
feat_imp_pd = pd.Series(feat_imp, index=features_list).sort_values(ascending=False)
print('\n')
print(feat_imp_pd.head(10))

# We save the model.
#pickle.dump(model,open('RF_model','wb'))



 



        
