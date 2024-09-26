# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:18:38 2024

@author: lopez
"""

# We import some useful libraries.
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float
import seaborn as sns
from scipy import stats

def get_iou(image, label):
    """This function calculates the IoU parameter for the segmented image
    (image) and the manually labelled one (label)."""
    overlap = np.sum(image & label)
    union = np.sum(image | label)
    return overlap / union

def get_dice(image, label):
    """This function calculates the Dice score for the segmented image
    (image) and the manually labelled one (label)."""
    overlap = np.sum(image & label)
    suma = np.sum(image) + np.sum(label)
    return (2*overlap) / suma

def get_accuracy(image, label):
    """This function calculates the pixel per pixel accuracy."""
    image_inv = np.invert(image)
    label_inv = np.invert(label)
    TP = np.sum(image & label)
    TN = np.sum(image_inv & label_inv)
    FP = np.sum(image & label_inv)
    FN = np.sum(label & image_inv)
    return (TP + TN) / (TP + TN + FP + FN)

def get_precision(image, label):
    """This function calculates the pixel per pixel precision."""
    label_inv = np.invert(label)
    TP = np.sum(image & label)
    FP = np.sum(image & label_inv)
    return TP / (TP + FP)

def get_recall(image, label):
    """This function calculates the pixel per pixel recall."""
    image_inv = np.invert(image)
    TP = np.sum(image & label)
    FN = np.sum(label & image_inv)
    return TP / (TP + FN)

def get_f1_score(image, label):
    """This function calculates the pixel per pixel F1 score."""
    precision = get_precision(image, label)
    recall = get_recall(image, label)
    return 2 * ((precision * recall) / (precision + recall))

def get_eval_params(string1, string2, label_string1, label_string2, num1 = 1, num2 = 34):
    """This function generates all the evaluation parameters."""
    
    # We create a list of evaluation parameters.
    iou_list = []
    dice_list = []
    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []

    # We iterate throught the whole set of images.    
    for i in np.arange(num1,num2,1):
        # We import the images.
        img = io.imread(string1+str(i)+string2) > 0.5
        # We import the labels.
        label = img_as_float(io.imread(label_string1+str(i)+label_string2))[2:62, 2:62] > 0.5
        # We calculate the IoU.
        iou = get_iou(img, label)
        # We append it to the list.
        iou_list.append(iou)
        # We calculate the Dice score.
        dice = get_dice(img, label)
        # We append it to the list.
        dice_list.append(dice)
        # We calculate the accuracy.
        acc = get_accuracy(img, label)
        # We append it to the list.
        acc_list.append(acc)
        # We calculate the precision.
        pre = get_precision(img, label)
        # We append it to the list.
        pre_list.append(pre)
        # We calculate the recall.
        rec = get_recall(img, label)
        # We append it to the list.
        rec_list.append(rec)
        # We calculate the F1 score.
        f1 = get_f1_score(img, label)
        # We append it to the list.
        f1_list.append(f1)
        
    return iou_list, dice_list, acc_list, pre_list, rec_list, f1_list

def plot_violin_box(list_data):
    """This function draws a violin plot, a box plot, and a swarm plot
    for the data contained in list_data."""
    fig, ax = plt.subplots(1,3,figsize=(15,4))
    ax[0].violinplot(list_data)
    ax[1].boxplot(list_data)
    sns.swarmplot(list_data, ax=ax[2])
    plt.tight_layout()
    plt.show()
    
def plot_grid(list_data, min_ax=0.90, max_ax=1.0, label='IoU', text_height=0.991):
    """This function creates a violin plot, a box plot, and a swarm plot
    for the data contained in list_data. It does it by reorganizing
    the plots."""
    colors = ['blue', 'orange', 'green']
    fig_1 = plt.figure(constrained_layout=True, figsize=(16,10))
    gs = fig_1.add_gridspec(2, 8)
    f1_ax1 = fig_1.add_subplot(gs[0, 0:3])
    violins = f1_ax1.violinplot(list_data)
    for ind, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[ind])
        pc.set_edgecolor(colors[ind])
    f1_ax1.set_ylim(min_ax,max_ax)
    f1_ax1.set_ylabel(label, size=24)
    f1_ax1.text(0.70, text_height,'K-means', fontsize=20, color='blue')
    f1_ax1.text(1.92, text_height,'RF', fontsize=20, color='orange')
    f1_ax1.text(2.93, text_height,'Li', fontsize=20, color='green')
    f1_ax1.set_xticks([])
    f1_ax1.tick_params(axis='both', which='major', labelsize=14)
    f1_ax2 = fig_1.add_subplot(gs[0,4:-1])
    boxes_d = f1_ax2.boxplot(list_data, patch_artist=True)
    for ind, patch in enumerate(boxes_d['boxes']):
        patch.set_facecolor(colors[ind])
        patch.set_edgecolor(colors[ind])
    f1_ax2.text(0.70, text_height,'K-means', fontsize=20, color='blue')
    f1_ax2.text(1.92, text_height,'RF', fontsize=20, color='orange')
    f1_ax2.text(2.93, text_height,'Li', fontsize=20, color='green')
    f1_ax2.set_ylim(min_ax, max_ax)
    f1_ax2.set_ylabel(label, size=24)
    f1_ax2.set_xticks([])
    f1_ax2.tick_params(axis='both', which='major', labelsize=14)
    f1_ax3 = fig_1.add_subplot(gs[1, 2:5])
    sns.swarmplot(list_data, ax=f1_ax3)
    f1_ax3.text(-0.35, text_height,'K-means', fontsize=20, color='blue')
    f1_ax3.text(0.9, text_height,'RF', fontsize=20, color='orange')
    f1_ax3.text(1.9, text_height,'Li', fontsize=20, color='green')
    f1_ax3.set_ylim(min_ax, max_ax)
    f1_ax3.set_ylabel(label, size=24)
    f1_ax3.set_xticks([])
    f1_ax3.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(label+'.png', dpi=300)
    plt.show()
    

# We get all the parameters for the K-means segmentation.
string1 = 'C:/Users/lopez/Documents/Desktop old PC/AI MSC/AI Capstone Proposal/Intensity images/Kmeans results/Pollen_'
string2 = '_Kmeans.tif'
label_string1 = 'C:/Users/lopez/Documents/Desktop old PC/AI MSC/AI Capstone Proposal/Intensity images/Labels/Pollen_'
label_string2 = '_label.tif'
num1 = 1 
num2 = 34
iou_kmeans, dice_kmeans, acc_kmeans, pre_kmeans, rec_kmeans, f1_kmeans = get_eval_params(string1=string1, string2=string2, label_string1=label_string1, label_string2=label_string2, num1=num1, num2=num2)

# We get all the parameters for the RF segmentation.
string1 = 'C:/Users/lopez/Documents/Desktop old PC/AI MSC/AI Capstone Proposal/Intensity images/RF results/Pollen_'
string2 = '_RF_segmented.tif'
label_string1 = 'C:/Users/lopez/Documents/Desktop old PC/AI MSC/AI Capstone Proposal/Intensity images/Labels/Pollen_'
label_string2 = '_label.tif'
num1 = 2 
mun2 = 34
iou_RF, dice_RF, acc_RF, pre_RF, rec_RF, f1_RF = get_eval_params(string1=string1, string2=string2, label_string1=label_string1, label_string2=label_string2, num1=num1, num2=num2)
    
# We get all the parameters for the Li segmentation.
string1 = 'C:/Users/lopez/Documents/Desktop old PC/AI MSC/AI Capstone Proposal/Intensity images/Li segmentation/Pollen_'
string2 = '_Li_segmented.tif'
label_string1 = 'C:/Users/lopez/Documents/Desktop old PC/AI MSC/AI Capstone Proposal/Intensity images/Labels/Pollen_'
label_string2 = '_label.tif'
num1 = 1 
mun2 = 34
iou_Li, dice_Li, acc_Li, pre_Li, rec_Li, f1_Li = get_eval_params(string1=string1, string2=string2, label_string1=label_string1, label_string2=label_string2, num1=num1, num2=num2)
    
# We organise the data for plotting.
iou = list((iou_kmeans, iou_RF, iou_Li))
dice = list((dice_kmeans, dice_RF, dice_Li))
acc = list((acc_kmeans, acc_RF, acc_Li))
pre = list((pre_kmeans, pre_RF, pre_Li))
rec = list((rec_kmeans, rec_RF, rec_Li))
f1 = list((f1_kmeans, f1_RF, f1_Li))

# We plot them.
plot_grid(iou, min_ax=0.88, max_ax=1.02, text_height=0.985)
plot_grid(dice, min_ax=0.94, max_ax=1.02, label='Dice score')
plot_grid(acc, min_ax=0.95, max_ax=1.02, label='Accuracy')
plot_grid(pre, min_ax=0.90, max_ax=1.02, text_height=1.005, label='Precision')
plot_grid(rec, min_ax=0.88, max_ax=1.02, text_height=1.005, label='Recall')
plot_grid(f1, min_ax=0.94, max_ax=1.02, label='F1 score')

# We carry out the t-tests.
iou_RF_vs_kmeans = stats.ttest_ind(iou_RF, iou_kmeans, equal_var=False)[1]
iou_RF_vs_Li = stats.ttest_ind(iou_RF, iou_Li, equal_var=False)[1]
dice_RF_vs_kmeans = stats.ttest_ind(dice_RF, dice_kmeans, equal_var=False)[1]
dice_RF_vs_Li = stats.ttest_ind(dice_RF, dice_Li, equal_var=False)[1]
acc_RF_vs_kmeans = stats.ttest_ind(acc_RF, acc_kmeans, equal_var=False)[1]
acc_RF_vs_Li = stats.ttest_ind(acc_RF, acc_Li, equal_var=False)[1]
pre_RF_vs_kmeans = stats.ttest_ind(pre_RF, pre_kmeans, equal_var=False)[1]
pre_RF_vs_Li = stats.ttest_ind(pre_RF, pre_Li, equal_var=False)[1]
rec_RF_vs_kmeans = stats.ttest_ind(rec_RF, rec_kmeans, equal_var=False)[1]
rec_RF_vs_Li = stats.ttest_ind(rec_RF, rec_Li, equal_var=False)[1]
f1_RF_vs_kmeans = stats.ttest_ind(f1_RF, f1_kmeans, equal_var=False)[1]
f1_RF_vs_Li = stats.ttest_ind(f1_RF, f1_Li, equal_var=False)[1]

# Print.
print('Precision and recall are statistically significant.')