# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:36:46 2020

@author: lopez
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy.fftpack import fft, ifft
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
from lmfit import Model, Parameter
from sklearn import cluster
from scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns
from scipy import signal
import tqdm
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from sdtfile import SdtFile
import pathlib
import keras
import tensorflow as tf
import pickle
from skimage import io, img_as_float, filters, feature, measure, segmentation


class IO_FLIM:
    """This class opens an image and contains functions to analyse such an image."""
    def __init__(self, filename):
        """Reads PicoQuant/Leica FLIM images stored as .bin files or Becker & Hickl
        FLIM images stored as .std files and outputs a dictionary with the most relevant
        metadata (see below), an image shaped as (y,x,t) where y is the number of Y
        pixels, x is the number of X pixels and t is the number of bins in the TCSPC
        histogram, a TCSPC decay of the whole image as a list that contains two
        numpy arrays: the first one is the time axis in ns and the second one is
        the intensity axis of the TCSPC histogram..
    
        Contents of the dictionary:
        
        PixX = number of X axis pixels,
        PixY = number of Y axis pixels,
        PixResol = pixel resolution in micrometers, this is only available for .bin images.
        TCSPCChannels = number of bins in the TCSPC decay,
        TimeResol = time resolution in ns.
        Frequency = the repetition frequency of the laser (in 1/seconds).
        """
        if pathlib.Path(filename).suffix == '.bin':
            self.collection = {}
            self.collection[0] = OpenBin(filename)
            self.num_images = 1
            
        elif pathlib.Path(filename).suffix == '.sdt':
            self.collection = {}
            sdt = SdtFile(filename)
            self.num_images = len(sdt.data)
            for i in range(self.num_images):
                self.collection[i] = OpenSdt(sdt,i)
            def __getitem__(self, item):
                return self.collection[item]
               
        
    def __get_decay__(self):
        """The input must be an image with shape (y,x,t), i.e. a cube. The output is a list
        that contains two numpy arrays: the first one is the time axis in ns and the second
        one is the intensity axis of the TCSPC histogram. You can plot the histogram by
        typing, for example, matplotlib.pyplot.plot(results[0],results[1])."""
        intensities = []
        times = np.linspace(0,self.dictionary['TCSPCChannels']*self.dictionary['TimeResol'],self.dictionary['TCSPCChannels'])
        for t in range(self.image.shape[2]):
            value = np.sum(self.image[:,:,t]) # Here the pixels used for the calculation of the TCSPC histogram can be selected in some way. 
            intensities.append(value)
        results = []
        results.append(times)
        results.append(np.array(intensities))
        return results
    
    def __estimate_IRF__(self):
        """This function estimates the IRF from the decay of the image i.e., self.decay 
        obtained by the __get_decay__() function). The output is a list that contains two
        numpy arrays: the first one is the time axis in ns and the second one is the
        intensity axis of the IRF."""
        def exp_model(x, ampl1=1.0, tau1=0.1):             
            exponential = ampl1*np.exp(-x/tau1)
            return exponential
        def calculate_new_y_value(yvalue,time_resolution,tau1):
            new_y_value = yvalue*np.exp(-time_resolution/tau1)
            return new_y_value
        time_resolution = self.decay[0][1] - self.decay[0][0]
        index = np.argmax(self.decay[1])
        dec_max = self.decay[1][index+3:index+8] #dec_max = decay[1][index+3:index+8]
        dec_max_x = self.decay[0][index+3:index+8] #dec_max_x = decay[0][index+3:index+8]       
        dec_model = Model(exp_model, nan_policy='propagate')
        res = dec_model.fit(dec_max, x=dec_max_x, ampl1=dec_max[0])
        IRF = []
        for i in range(len(self.decay[1])-1):
            new_y = calculate_new_y_value(self.decay[1][i],time_resolution,tau1=res.params['tau1'].value)
            difference = np.abs(self.decay[1][i+1]-new_y)
            IRF.append(difference)
        IRF = signal.savgol_filter(IRF, 11, 2)
        IRF = np.append(IRF,0)
        IRF = ndi.shift(IRF,1)
        max_IRF = np.argmax(IRF)
        for count,value in enumerate(IRF[max_IRF:]):
            if value < 0.2*np.max(IRF):
                IRF[max_IRF+count] = 0
        for count,value in enumerate(IRF[:max_IRF]):
            if value < 0.01*np.max(IRF):
                IRF[count] = 0
        IRF[:max_IRF-int((len(IRF)*0.01))] = 0 #IRF[:max_IRF-50] = 0
        IRF_results = []
        IRF_results.append(self.decay[0])
        IRF_results.append(IRF)
        return IRF_results
    
    def get_int(self):
        """The input must be an image  with shape (y,x,t)), a.k.a. a cube. The output
        is an image with the same x,y shape as the input image and that contains the
        total fluorescence intensity at each pixel."""
        int_img = np.zeros((self.image.shape[0],self.image.shape[1]))
        print('Obtaining the intensity image...')
        for y in tqdm.tqdm(range(self.image.shape[0])):
            for x in range(self.image.shape[1]):
                list_intensities = []
                for t in range(self.image.shape[2]):
                    value = self.image[y,x,t]
                    list_intensities.append(value)
                int_img[y,x] = np.sum(list_intensities)
        return int_img
    
    def get_FLIM(self, thresh=10):
        """The input must be an image with shape (y,x,t)) (the one created when
        instantiating the IO_FLIM class). The output is an image 
        with the same x,y shape as the input image and that contains the average
        lifetime at each pixel. The average lifetime is calculated as the weighted
        average of the TCSPC decay and is somewhat equivalent to what PicoQuant calls
        "Fast FLIM. The thresh paramater, with a default value of 100, means that pixels with
        less than the chosen number of photons will be assigned a lifetime value 
        of 0."""
        # This first part is just to find the maximum of the TCSPC decay of the 
        # whole image and then generate the corresponding time data.
        TCSPC = self.decay # Gets the decay of the whole image.
        times = TCSPC[0] # Gets the time data.
        decay = TCSPC[1] # Gets the intensity data.
        time_resolution = times[1] - times[0] # Time resolution in seconds
        index = np.where(decay == np.max(decay))[0][0] # Finds the index of the peak of the decay.
        decay = decay[index:] # Eliminates all the bins before the peak.
        times = np.linspace(0,len(decay)*time_resolution,len(decay)) # This is the right time axis.
        # Now the actual FAST FLIM image calculation begins.
        fast_flim_img = np.zeros((self.image.shape[0],self.image.shape[1]))
        print('Obtaining the fast FLIM image...')
        for y in tqdm.tqdm(range(self.image.shape[0])):
            for x in range(self.image.shape[1]):
                intensities = []
                for t in range(self.image.shape[2]):
                    intensities.append(self.image[y,x,t])
                if np.sum(intensities) >= thresh:
                    intensities = intensities[index:] # Gets rid of the bins before the peak.
                    avg = (np.sum(times*intensities))/np.sum(intensities)
                    fast_flim_img[y,x] = avg
                else:
                    fast_flim_img[y,x] = 0
        return fast_flim_img
    
    def get_phasor(self, thresh=10):
        """This function calculates the phasor plot. All pixels with less photons
        than the value specified by "thresh" (with a default value of 10) will be
        disregarded. The output is a list containing three numpy arrays. The first of these arrays
        contains the g values and the second the s values. The third of the arrays contains
        the pixel locations (x,y)."""
        TCSPC = self.decay # Gets the decay of the whole image.
        times = TCSPC[0] # Gets the time data.
        times = times/1E9 # Transforms the time information from nanoseconds to seconds.
        def corr_phasor(g_exp,s_exp,g_IRF,s_IRF,w):
            """This equation corrects the g and s values using the g and s values of the IRF. The inputs are g_exp and s_exp (the uncorrected g and s values)
            and g_IRF and s_IRF (the g and s values of the IRF)."""
            matrices = (1/((g_IRF**2)+(s_IRF**2))) * np.matmul(np.array([[g_IRF,s_IRF],[-s_IRF,g_IRF]]),np.array([[g_exp],[s_exp]]))
            g_corr = matrices[0][0]
            s_corr = matrices[1][0]
            return g_corr,s_corr
        def calc_g_s(decay,time,w):
            """This equation calculates the g and s values. Inputs are the decay (1-dimensional intensity values), time (1-dimensional time values)
            and w (the angular frequency of the laser excitation = 2 x pi x laser frequency in seconds to the minus 1)."""
            g = np.sum(decay*np.cos(w*time))/np.sum(decay)
            s = np.sum(decay*np.sin(w*time))/np.sum(decay)
            return g,s
        w = self.dictionary['Frequency']*2*np.pi # The angular frequency of laser excitation.
        # We get the IRF g and s values.
        g_IRF,s_IRF = calc_g_s(self.IRF[1],times,w)
        # Here we calculate the g and s values for every pixel in the image correcting each one of them in turn.
        # Now the actual phasor calculation begins.
        w = self.dictionary['Frequency']*2*np.pi # The angular frequency of laser excitation.
        g = []
        s = []
        pix = []
        print('Obtaining the phasor plot...')
        for y in tqdm.tqdm(range(self.image.shape[0])):
            for x in range(self.image.shape[1]):
                intensities = []
                for t in range(self.image.shape[2]):
                    intensities.append(self.image[y,x,t])
                if np.sum(intensities) >= thresh:
                    g_local,s_local = calc_g_s(intensities,times,w)
                    g_corr,s_corr = corr_phasor(g_local,s_local,g_IRF,s_IRF,w)
                    g.append(g_corr) 
                    s.append(s_corr) 
                    pix.append((x,y))
                else:
                    pass
        phasor = [[],[],[]]
        phasor[0] = g
        phasor[1] = s
        phasor[2] = pix
        return phasor

class OpenSdt(IO_FLIM):
    """This class is just an auxiliary class for IO_FLIM. It shouldn't be instantiated directly. It's just to be used within IO_FLIM.
    The input is an open sdt file and a position within the file. The output is an object that contains all the attributes common to IO_FLIM objects."""
    def __init__(self,sdt,position):
        dictionary = {}
        dictionary['PixX'] = (sdt.measure_info[position].image_x[0])+1
        dictionary['PixY'] = sdt.measure_info[position].image_y[0]
        dictionary['TCSPCChannels'] = sdt.times[position].shape[0]
        dictionary['TimeResol'] = sdt.times[position][1] / 1e-9 # time resolution in ns
        dictionary['Frequency'] = 1/(1e-9*dictionary['TCSPCChannels']*dictionary['TimeResol'])
        img_cube = np.reshape(sdt.data[position],(dictionary['PixY'],dictionary['PixX'],dictionary['TCSPCChannels']))
        
        self.dictionary = dictionary
        self.image = img_cube
        self.image = img_cube
        self.dictionary = dictionary
        self.decay = self.__get_decay__()
        self.IRF = self.__estimate_IRF__()
        
class OpenBin(IO_FLIM):
    """This class is just an auxiliary class for IO_FLIM. It shouldn't be instantiated directly. It's just to be used within IO_FLIM.
    the input is an .bin file and. the output is an object that contains all of the attributes common to IO_FLIM objects."""
    def __init__(self, filename):
        with open(filename, 'rb') as file:
            dictionary = {} # Creates an empty dictionary.
            # pixels in X-direction (int32)
            PixX = struct.unpack('i',file.read(4))[0]
            dictionary['PixX'] = PixX
            # pixels in Y-direction (int32)
            PixY = struct.unpack('i',file.read(4))[0]
            dictionary['PixY'] = PixY
            # spatial pixel resolution in micrometer (float32)
            PixResol = struct.unpack('f',file.read(4))[0]
            dictionary['PixResol'] = PixResol
            # number of TCSPC channels per pixel (float32)
            TCSPCChannels = struct.unpack('i',file.read(4))[0]
            dictionary['TCSPCChannels'] = TCSPCChannels
            # time resolution of the TCSPC histograms in ns (float32)
            TimeResol = struct.unpack('f',file.read(4))[0]
            dictionary['TimeResol'] = TimeResol
            # adds the repetition frequency of the laser (in seconds) to the dictionary.
            dictionary['Frequency'] =1/(1e-9*dictionary['TCSPCChannels']*dictionary['TimeResol'])

            # the histogram is composed of TCSPCChannels int32 values
            DecayFormatString = str(TCSPCChannels)+'i'

            img_cube = np.zeros((PixY,PixX,TCSPCChannels))
            
            print('Reading the file...')
            for y in tqdm.tqdm(range(PixY)):
                for x in range(PixX):
                    TCSPCDecay=struct.unpack(DecayFormatString,file.read(TCSPCChannels*4))
                    # Assigns the TCSPCDecay of this specific pixel to a voxel in the cube.
                    for t in range(len(TCSPCDecay)):
                        img_cube[y,x,t] = TCSPCDecay[t]
            
            self.dictionary = dictionary
            self.image = img_cube
            self.dictionary = dictionary
            self.decay = self.__get_decay__()
            self.IRF = self.__estimate_IRF__()
            
    
class FLIM_methods:
    """This class contains a collection of methods needed to manipulate FLIM images."""
    @staticmethod
    def display_phasor(phasor, frequency, nbins=300, clusters=0, savename=None, dpi=600):
        """This functions displays a phasor plot. The inputs are the phasor, which
        must be a list containing two (or three) numpy arrays: the first must contain the g values
        and the second must contain the s values. This is the output format produced by the
        'obtain_phasor' function. If savename is set to any string whatsoever, the image
        will be saved using that string as the filename."""
        print('Preparing the phasor plot. This may take a few minutes...')
        nbins=nbins
        k = gaussian_kde([phasor[0],phasor[1]])
        xi, yi = np.mgrid[0:1:nbins*1j,0:1:nbins*1j]
        zi = k(np.vstack([xi.flatten(),yi.flatten()]))
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.hot)
        x = np.arange(0.0,1.01,0.01,dtype=np.float64)
        y = np.sqrt((0.5*0.5)-((x-0.5)*(x-0.5)))
        ax.plot(x,y,'w--')
        ax.axis([0,1,0,0.55])
        ax.set_ylabel('S', fontsize=16)
        ax.set_xlabel('G', fontsize=16)
        ax.tick_params(labelsize=12, width=3)
        if clusters >= 1:
            # This gets the centroids of the scatterplot.
            X = np.array(list(zip(phasor[0], phasor[1]))).reshape(len(phasor[0]), 2)
            kmeans_model = KMeans(n_clusters=clusters, n_init=10).fit(X)
            centers = np.array(kmeans_model.cluster_centers_)
            plt.scatter(centers[:,0], centers[:,1], marker="o", color='cyan',s=40,edgecolors='blue') # This plots the centroids.
            # Here we obtain the lifetimes of the centroids.
            lifetimes = []
            for i in range(clusters):
                lifetime = (centers[i][1]/centers[i][0])*(1/(np.pi*2*frequency))
                lifetimes.append(lifetime)
        if savename is not None:
            plt.savefig(str(savename),dpi=dpi)
        plt.show()
        if clusters >= 1:
            return lifetimes
        
    @staticmethod
    def import_IRF(filename, image):
        """This function imports an IRF from a .bin file. The input is the .bin filename and the IO_FLIM image object that is to be
        used in combination with this IRF. That latter is necessary to check if the IRF and the decay are compatible with each other."""
        with open(filename, 'rb') as file:
            # pixels in X-direction (int32)
            PixX = struct.unpack('i',file.read(4))[0]
            # pixels in Y-direction (int32)
            PixY = struct.unpack('i',file.read(4))[0]
            # spatial pixel resolution in micrometer (float32)
            PixResol = struct.unpack('f',file.read(4))[0]
            # number of TCSPC channels per pixel (float32)
            TCSPCChannels = struct.unpack('i',file.read(4))[0]
            # time resolution of the TCSPC histograms in ns (float32)
            TimeResol = struct.unpack('f',file.read(4))[0]
            # repetition frequency of the laser (in seconds) to the dictionary.
            rep_rate =1/(1e-9*TCSPCChannels*TimeResol)
            # the histogram is composed of TCSPCChannels int32 values
            DecayFormatString = str(TCSPCChannels)+'i'

            img_cube = np.zeros((PixY,PixX,TCSPCChannels))
                
            print('Reading the IRF file...')
            for y in tqdm.tqdm(range(PixY)):
                for x in range(PixX):
                    TCSPCDecay=struct.unpack(DecayFormatString,file.read(TCSPCChannels*4))
                    # Assigns the TCSPCDecay of this specific pixel to a voxel in the cube.
                    for t in range(len(TCSPCDecay)):
                        img_cube[y,x,t] = TCSPCDecay[t]
                
            # We now extract the IRF from the whole image.
            intensities = []
            times = np.linspace(0,TCSPCChannels*TimeResol,TCSPCChannels)
            for t in range(img_cube.shape[2]):
                value = np.sum(img_cube[:,:,t]) # Here the pixels used for the calculation of the TCSPC histogram can be selected in some way. 
                intensities.append(value)
            results = []
            if len(times) != len(image.decay[0]):
                try:
                    f = interp1d(times,np.array(intensities))
                    new_intensities = f(image.decay[0])
                    results.append(image.decay[0])
                    results.append(new_intensities)
                    return results
                except:
                    print('The IRF and the decay have incompatible time resolutions.')
            else:
                results.append(times)
                results.append(np.array(intensities))
                return results
        
    @staticmethod    
    def display_FLIM(intensity_image,flim_image,lifetime_bar_lims=None,intensity_bar_lims=None,scalebar=False,resolution=None,save_file_name=None):
        """This function displays the FLIM image. The colours in the FLIM image correspond to lifetime values. The brightness in the image corresponds
        to intensity values. There are several optional parameters:
            lifetime_bar_lims determines the limits of the lifetime bar. It should be entered as a tuple (e.g., lifetime_bar_lims=(1,3))
            intensity_bar_lims determines the limits of the intensity bar. It should be entered as a tuple (e.g., intensity_bar_lims=(0,800))
            To draw a scalebar on the image, the parameters scalebar should be set to True (e.g., scalebar=True) AND the pixel size of the image in micrometers should be provided (e.g., resolution = img.collection[0].dictionary['PixResol'])
            The image can be saved if save_file_name is set to a filename (e.g., save_file_name='image.png')"""
        if lifetime_bar_lims is not None:
            flim_image = np.clip(np.copy(flim_image),lifetime_bar_lims[0],lifetime_bar_lims[1])
        if intensity_bar_lims is not None:
            intensity_image = np.clip(np.copy(intensity_image),intensity_bar_lims[0],intensity_bar_lims[1])
        mask = Normalize()(intensity_image)
        fig, ax = plt.subplots(figsize=(10,10))
        im1 = ax.imshow(intensity_image, cmap='gray')
        im2 = ax.imshow(flim_image, cmap='jet', alpha=mask)
        ax.axis('off')
        pos = ax.get_position()
        bar_h = (pos.y1 - pos.y0) * 0.4  # 0.5 joins the two bars, e.g. 0.48 separates them a bit
        ax_cbar1 = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.03, bar_h])
        ax_cbar1.grid(False)
        cbar1 = fig.colorbar(im1, cax=ax_cbar1, orientation='vertical')
        ax_cbar2 = fig.add_axes([pos.x1 + 0.02, pos.y1 - bar_h, 0.03, bar_h])
        ax_cbar2.grid(False)
        cbar2 = fig.colorbar(im2, cax=ax_cbar2, orientation='vertical')
        if scalebar is True and resolution is not None:
            fontprops = fm.FontProperties(size=20)
            scalebar = AnchoredSizeBar(ax.transData, intensity_image.shape[0]/5, str(int(intensity_image.shape[0]/5*resolution))+' μm', 'lower left', pad=1, color='white', frameon=False, size_vertical=intensity_image.shape[0]/100, fontproperties=fontprops)
            ax.add_artist(scalebar)
        if save_file_name is not None:
            plt.savefig(save_file_name, dpi=600, bbox_inches='tight')
        plt.show()

    @staticmethod 
    def get_lifetime_distribution(flim_image, bins=100, range=(0.1,20)):
        """Calculates the lifetime histogram of a flim image. It returns the
        histogram. Range starts at 0.1 to avoid the 0 values present in thresholded 
        images."""
        hist = plt.hist(flim_image.flat, bins=bins, range=range)
        return hist
    
    @staticmethod
    def estimate_IRF_NN(decay):
        """This function estimates the IRF using a neural network. The IRF will consist of 512 bins with a 
        temporal resolution of 0.09696969 ns per bin. This should be the temporal resolution of the decay too."""
        decay = decay / np.max(decay) # First we normalize the decay.
        length = len(decay)
        if length < 512: # If the decay has fewer than 512 bins, we extend it to 512 bins.
            mn = np.mean(decay[-2:])
            diff = length - 512
            mn = [mn]*diff
            decay = list(decay) + mn
            decay = np.array(decay)
        elif length > 512: # If the decay has more than 512 bins, we eliminate bins and keep only the first 512.
            diff = length - 512
            decay = decay[:-diff]
        decay = decay[None,:]
        model = tf.keras.models.load_model("D:/Microscope users/Sergio Lopez/Python/AI Capstone/Extract IRF/Model 1/Learning rate 0.0001/Model_1_IRF_extraction_lr00001.keras")
        irf = model.predict(decay)
        irf = irf[0,:]
        aver = np.mean(irf[-2:]) # This and subsequent steps are used to extend the IRF so as to reach 512 bins.
        aver = [aver]*462
        irf = list(irf) + aver
        irf = np.array(irf)
        return irf
    
    @staticmethod  
    def extract_lifetime_NN(decay):
        """This function extracts lifetime information from a decay using a neural network. The decay should
        have 512 bins and a temporal resolution of 0.09696969 ns per bin. So, decays will be shortened to 512 bins
        The function returns an array that contains the mean value of the lifetime distribution, t, in 
        position [0] and the heterogeneity parameter, q, in position [1]. These two parameters are sufficient 
        to describe a fluoreascence decay in the context of a power-like model (Włodarczyk et al. 
        Interpretation of Fluorescence Decays using a Power-like Model. Biophys. J. 85 (2003) 589.)"""
        decay = decay / np.max(decay) # First we normalize the decay.
        length = len(decay)
        if length < 512: # If the decay has fewer than 512 bins, we extend it to 512 bins.
            mn = np.mean(decay[-2:])
            diff = length - 512
            mn = [mn]*diff
            decay = list(decay) + mn
            decay = np.array(decay)
        elif length > 512: # If the decay has more than 512 bins, we eliminate bins and keep only the first 512.
            diff = length - 512
            decay = decay[:-diff]
        decay = decay[None,:]
        model = tf.keras.models.load_model("D:/Microscope users/Sergio Lopez/Python/AI Capstone/Fit Decay/Model 3/Learning rate 0.0005/Model_3_fit_lr00005.keras")
        values = model.predict(decay)
        return values

    @staticmethod
    def display_lifetime_distribution(fast_flim_img, font_scale=1.5, axes_facecolor='white', figure_facecolor='white', savename=None, dpi=600, xlim=None):
        """This function displays the lifetime distribution of a FLIM image. The input
        is a lifetime image (e.g., a fast FLIM image). You can save the figure is you
        change 'savename' from None to a string. This string will become the filename
        of the figure that will be saved. If you want to set x-axis limits just change
        the 'xlim' parameter to something like, for example xlim=(0.0,5.0)"""
        if xlim is not None:
            lifetime_values = fast_flim_img.flatten()
            df_FLIM = pd.DataFrame(lifetime_values, columns=['Lifetimes'])
            df_FLIM = df_FLIM[df_FLIM.Lifetimes != 0]
            sns.set(font_scale=1.5, rc={'figure.facecolor':'white'})
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
            sns.boxplot(data=df_FLIM, x="Lifetimes", ax=ax_box, color='darkorange')
            sns.histplot(data=df_FLIM, x="Lifetimes", ax=ax_hist, color='darkorange', kde=True, stat='density')
            ax_box.set(xlabel='')
            ax_box.set_xlim(xlim)
            plt.tight_layout()
            if savename is not None:
                plt.savefig(str(savename),dpi=dpi)
            plt.show()
        else:
            lifetime_values = fast_flim_img.flatten()
            df_FLIM = pd.DataFrame(lifetime_values, columns=['Lifetimes'])
            df_FLIM = df_FLIM[df_FLIM.Lifetimes != 0]
            sns.set(font_scale=1.5, rc={'figure.facecolor':'white'})
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
            sns.boxplot(data=df_FLIM, x="Lifetimes", ax=ax_box, color='darkorange')
            sns.histplot(data=df_FLIM, x="Lifetimes", ax=ax_hist, color='darkorange', kde=True, stat='density')
            ax_box.set(xlabel='')
            plt.tight_layout()
            if savename is not None:
                plt.savefig(str(savename),dpi=dpi)
            plt.show()

    @staticmethod
    def fit_decay(IRF, decay, num_decay=1): # Note: try adding weights for the fitting to improve around the IRF. There are very few points around the IRF and the fit doesn't care about them.
        """Fits the overall decay of the image (or of an ROI within the image) to
        exponential equations with 1-4 terms (i.e., 1-4 lifetimes). the inputs are:
            IRF: list containing two arrays, the first one the time values and the second the intensity values.
            decay: list containing two arrays, the first one the time values and the second the intensity values.
            num_decay (optional): number of exponentials to use (1 by default).
        The output of fit_decay is an lmfit object that contains the residuals (get it 
        as object.residual), the report of the fit (get it as print(object.fit_report())), 
        the best fit to the data (get it as object.best_fit), and a plot of the best fit
        (get it as object.plot())."""
        x = IRF[0]
        IRF = IRF[1]
        decay = decay[1]
        def Exp(x, ampl1=1.0, tau1=3.0): # This generates an exponential model.
            res = ampl1*np.exp(-x/tau1)
            return res
        def Conv(IRF,decay): # This convolves a model with the data (data = Instrument Response Function, IRF).
            conv = ifft(fft(decay) * fft(IRF)).real
            return conv
        if int(num_decay) == 1: # If the user chooses to use a model equation with one exponential term.
            def fitting(x, ampl1=1.0, tau1=3.0, y0=0, x0=0): 
                IRF_shifted = ndi.shift(IRF,x0)
                exponential = Exp(x,ampl1,tau1)
                convolved = Conv(IRF_shifted,exponential)
                return convolved + y0
            modelling = Model(fitting, nan_policy='propagate')
            res = modelling.fit(decay,x=x, weights=1./(np.sqrt(decay)+1))
        if int(num_decay) == 2: # If the user chooses to use a model equation with two exponential terms.
            def fitting(x, ampl1=1.0, tau1=3.0, ampl2=1.0, tau2=1.0, y0=0, x0=0): 
                IRF_shifted = ndi.shift(IRF,x0)
                exponential = Exp(x,ampl1,tau1)+Exp(x,ampl2,tau2)
                convolved = Conv(IRF_shifted,exponential)
                return convolved + y0
            modelling = Model(fitting, nan_policy='propagate')
            res = modelling.fit(decay,x=x, weights=1./(np.sqrt(decay)+1))
        if int(num_decay) == 3: # If the user chooses to use a model equation with three exponential terms.
            def fitting(x, ampl1=1.0, tau1=3.0, ampl2=2.0, tau2=1.0, ampl3=3.0, tau3=5.0, y0=0, x0=0): 
                IRF_shifted = ndi.shift(IRF,x0)
                exponential = Exp(x,ampl1,tau1)+Exp(x,ampl2,tau2)+Exp(x,ampl3,tau3)
                convolved = Conv(IRF_shifted,exponential)
                return convolved + y0
            modelling = Model(fitting, nan_policy='propagate')
            res = modelling.fit(decay,x=x, weights=1./(np.sqrt(decay)+1))
        if int(num_decay) == 4: # If the user chooses to use a model equation with four exponential terms.
            def fitting(x, ampl1=1.0, tau1=3.0, ampl2=2.0, tau2=1.0, ampl3=0.5, tau3=0.1, ampl4=4.0, tau4=6.0, y0=0, x0=0):             
                IRF_shifted = ndi.shift(IRF,x0)
                exponential = Exp(x,ampl1,tau1)+Exp(x,ampl2,tau2)+Exp(x,ampl3,tau3)+Exp(x,ampl4,tau4)
                convolved = Conv(IRF_shifted,exponential)
                return convolved + y0
            modelling = Model(fitting, nan_policy='propagate')
            res = modelling.fit(decay,x=x, weights=1./(np.sqrt(decay)+1))
        return res
            
    @staticmethod
    def display_fit(IRF, decay, results, figsize=(8,6), xlim=None, ylim=None, fontsize=16, legend_fontsize=12, dpi=600, save=None, name_figure='test'):
        """Displays and saves (optional) a decay fit. The inputs are:
            xvalues: 1-dimensional array containing the time values of the decay and IRF (both should be the same).
            decay: list containing two arrays, the first one the time values and the second the intensity values.
            IRF: list containing two arrays, the first one the time values and the second the intensity values.
            results: lmfit object generated by the fit_decay() or fit_image() functions.
            figsize (optional): size of the figure.
            xlim (optional): limits of the x-axis of the figure.
            ylim (optional: limites of the y-axis of the decay plot.
            fontsize (optional): size of the y labels.
            legend_fontsize (optional): size of the font of the legends.
            dpi (optional): dpi of the saved figure.
            save (optional): change to 'yes' to save the image. It doesn't save the image by default.
            name_figure (optional): name of the figure that is saved."""
        xvalues = decay[0]
        decay = decay[1]
        IRF = IRF[1]
        best_fit = results.best_fit
        fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},figsize=figsize)
        l1, l2, l3 = a0.semilogy(xvalues[::3], decay[::3],'bo', xvalues, IRF, 'green', xvalues, best_fit, 'r-')
        if xlim:
            a0.set_xlim(xlim)
        if ylim:
            a0.set_ylim(ylim)
        a0.set_ylabel('Intensity / counts', fontsize=fontsize)
        a0.legend((l1, l2, l3), ('decay', 'IRF', 'fit'), loc='upper right', shadow=True, fontsize=legend_fontsize)
        a1.plot(xvalues, results.residual)
        a1.set_xlim(xlim)
        a1.set_xlabel('Time / ns', fontsize=fontsize)
        a1.set_ylabel('Residuals', fontsize=fontsize)
        fig.tight_layout()
        if save is not None:
            plt.savefig(name_figure+'.png', dpi=dpi)
        plt.show()
            
    @staticmethod        
    def k_means_segmentation(image, n_clusters=3):
        """This function clusters the pixels into the specified number of clusters
        (i.e., n_clusters). The input can be any greyscale image, such as a fast
        FLIM image, for example. The output is the segmented image."""
        img_flattened = image.reshape((-1,1))
        k_m = cluster.KMeans(n_clusters=n_clusters, n_init=4)
        k_m.fit(img_flattened)
        values = k_m.cluster_centers_.squeeze()
        indexes = []
        for i,j in enumerate(values):
            indexes.append(i)
        labels = k_m.labels_
        img_segm = np.choose(labels, indexes)
        img_segm.shape = image.shape
        # We keep only the largest object.
        labels = ndi.label(img_segm)[0]
        props = measure.regionprops(labels)
        areas = []
        for prop in props:
            areas.append((prop.area, prop.label))
        area_max = max(areas, key=lambda item: item[0])[0]
        for i in areas:
            if i[0] == area_max:
                target_label = i[1]
        labels_cleaned = np.zeros_like(image)
        labels_cleaned[labels == target_label] = 1
        return labels_cleaned
    
    @staticmethod
    def add_phasor(phasor1,phasor2):
        """This functions allows you to add two phasors. The input are the two 
        phasors and the output is a third phasor which is the sum of the first two."""
        phasor=[phasor1[0]+phasor2[0],phasor1[1]+phasor2[1],phasor1[2]+phasor2[2]]
        return phasor

    @staticmethod
    def extract_lifetimes(results, n_components=1):
        """Extracts the tau values and the amplitude values from the results file.
        The inputs must be a results file generated by the fit_decay() function
        and the number of lifetime components (by default set to 1). The output 
        is a dictionary with the amplitude and lifetime values."""
        if n_components == 1:
            ampl1 = results.params['ampl1'].value
            tau1 = results.params['tau1'].value
            dictionary = {}
            dictionary['ampl1'] = ampl1
            dictionary['tau1'] = tau1
        elif n_components == 2:
            ampl1 = results.params['ampl1'].value
            tau1 = results.params['tau1'].value
            ampl2 = results.params['ampl2'].value
            tau2 = results.params['tau2'].value
            dictionary = {}
            dictionary['ampl1'] = ampl1
            dictionary['tau1'] = tau1
            dictionary['ampl2'] = ampl2
            dictionary['tau2'] = tau2
        elif n_components == 3:
            ampl1 = results.params['ampl1'].value
            tau1 = results.params['tau1'].value
            ampl2 = results.params['ampl2'].value
            tau2 = results.params['tau2'].value
            ampl3 = results.params['ampl3'].value
            tau3 = results.params['tau3'].value
            dictionary = {}
            dictionary['ampl1'] = ampl1
            dictionary['tau1'] = tau1
            dictionary['ampl2'] = ampl2
            dictionary['tau2'] = tau2
            dictionary['ampl3'] = ampl3
            dictionary['tau3'] = tau3
        elif n_components == 4:
            ampl1 = results.params['ampl1'].value
            tau1 = results.params['tau1'].value
            ampl2 = results.params['ampl2'].value
            tau2 = results.params['tau2'].value
            ampl3 = results.params['ampl3'].value
            tau3 = results.params['tau3'].value
            ampl4 = results.params['ampl4'].value
            tau4 = results.params['tau4'].value
            dictionary = {}
            dictionary['ampl1'] = ampl1
            dictionary['tau1'] = tau1
            dictionary['ampl2'] = ampl2
            dictionary['tau2'] = tau2
            dictionary['ampl3'] = ampl3
            dictionary['tau3'] = tau3
            dictionary['ampl4'] = ampl4
            dictionary['tau4'] = tau4
        else:
            dictionary = {}
        return dictionary
    
    @staticmethod
    def fit_image_exp_decay(image, dictionary, IRF, thresh=100, tau1=3.0, tau2=None, tau3=None, tau4=None):
        times = np.linspace(0,dictionary['TCSPCChannels']*dictionary['TimeResol'],dictionary['TCSPCChannels'])
        tau_dictionary = {}
        IRF = IRF[1]
        def Exp(times, ampl1=1.0, tau1=3.0): # This generates an exponential model.
            res = ampl1*np.exp(-x/tau1)
            return res
        def Conv(IRF,decay): # This convolves a model with the data (data = Instrument Response Function, IRF).
            conv = ifft(fft(decay) * fft(IRF)).real
            return conv
        if  tau2 is not None and tau3 is not None and tau4 is not None:
            image_fit = np.zeros((image.shape[0],image.shape[1],4))
            tau_dictionary['tau1'] = tau1
            tau_dictionary['tau2'] = tau2
            tau_dictionary['tau3'] = tau3
            tau_dictionary['tau4'] = tau4
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    intensities = []
                    for t in range(image.shape[2]):
                        intensities.append(image[y,x,t])
                    if np.sum(intensities) >= thresh:
                        def fitting(times=times, IRF=IRF, ampl1=1.0, tau1=tau1, ampl2=1.0, tau2=tau2, ampl3=1.0, tau3=tau3, ampl4=1.0, tau4=tau4):
                            exponential = Exp(times,ampl1,tau1)+Exp(times,ampl2,tau2)+Exp(times,ampl3,tau3)+Exp(times,ampl4,tau4)
                            convolved = Conv(IRF,exponential)
                            return convolved
                        modelling = Model(fitting, independent_vars=['times','IRF'],nan_policy='propagate')
                        res = modelling.fit(intensities,times=times, IRF=IRF, tau1=Parameter('tau1',tau1,False),tau2=Parameter('tau2',tau2,False),tau3=Parameter('tau3',tau3,False),tau4=Parameter('tau4',tau4,False)) # The use of 'Parameter with the 'vary' value as False keeps the lifetimes fixed during the fitting.
                        dictionary = FLIM_methods.extract_lifetimes(res, 4)
                        ampl1 = dictionary['ampl1']
                        ampl2 = dictionary['ampl2']
                        ampl3 = dictionary['ampl3']
                        ampl4 = dictionary['ampl4']
                        sum_ampl = ampl1 + ampl2 + ampl2 + ampl4
                        norm_ampl1 = ampl1 / sum_ampl
                        norm_ampl2 = ampl2 / sum_ampl
                        norm_ampl3 = ampl3 / sum_ampl
                        norm_ampl4 = ampl4 / sum_ampl
                        image_fit[y,x,0] = norm_ampl1
                        image_fit[y,x,1] = norm_ampl2
                        image_fit[y,x,2] = norm_ampl3
                        image_fit[y,x,3] = norm_ampl4
                    else:
                        image_fit[y,x,0] = 0
                        image_fit[y,x,1] = 0
                        image_fit[y,x,2] = 0
                        image_fit[y,x,3] = 0
        return image_fit, tau_dictionary
    
    @staticmethod 
    def rf_segmenter(image):
        """This function loads a trained Random Forest model and uses it to segment 
        an image."""
        # We crop the image because the Random Forest model was trained with cropped images.
        image = image[2:62,2:62]
        
        # We normalize the image.
        image = image / np.max(image)
        
        # We load the model.
        model = pickle.load(open('D:/Microscope users/Sergio Lopez/Python/AI Capstone/Segmentation model/RF_model', 'rb'))
        
        # We create an empty DataFrame.
        data = pd.DataFrame()
    
        # We add the original image as the first feature.
        orig_flat = image.reshape(-1)
    
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
                    convolved = ndi.convolve(image, kernel, mode='wrap')
                    convolved_flat = convolved.reshape(-1)
                    data[gabor_label] = convolved_flat
                    num += 1
    
        # We add Canny filters to the features.
        num = 1
        for sigma in np.arange(1,9,2):
            canny_label = 'Canny ' + str(num)
            cannied = feature.canny(image, sigma=sigma)
            cannied_flat = cannied.reshape(-1)
            data[canny_label] = cannied_flat*1.
            num += 1
            
        # We add a Sobel filter to the features.
        sobel = filters.sobel(image)
        sobel_flat = sobel.reshape(-1)
        data['Sobel'] = sobel_flat
    
        # We add a Prewitt filter to the features.
        prewitt = filters.prewitt(image)
        prewitt_flat = prewitt.reshape(-1)
        data['Prewitt'] = prewitt_flat
    
        # We add a Scharr filter to the features.
        scharr = filters.scharr(image)
        scharr_flat = scharr.reshape(-1)
        data['Scharr'] = scharr_flat
    
        # We add a Roberts filter to the features.
        roberts = filters.roberts(image)
        roberts_flat = roberts.reshape(-1)
        data['Roberts'] = roberts_flat
    
        # We add a few Gaussian filters to the features.
        num = 1
        for sigma in (1,2,3,4,5,6,7,8,9,10):
            gaussian_label = 'Gaussian ' + str(num)
            gaussian_img = filters.gaussian(image, sigma=sigma)
            gaussian_flat = gaussian_img.reshape(-1)
            data[gaussian_label] = gaussian_flat
            num += 1
            
        # We create the X data for the model to carry out the prediction.
        X = data.values
        
        # We carry out the prediction.
        results = model.predict(X)
        
        # We reshape the prediction.
        segmented = results.reshape((image.shape))
        
        # We keep only the largest object.
        labels = ndi.label(segmented)[0]
        props = measure .regionprops(labels)
        areas = []
        for prop in props:
            areas.append((prop.area, prop.label))
        area_max = np.max(areas)
        for i in areas:
            if i[0] == area_max:
                target_label = i[1]
        labels_cleaned = np.zeros_like(segmented)
        labels_cleaned[labels == target_label] = 1
        
        return labels_cleaned
    
    @staticmethod  
    def li_segmenter(image):
        """This function uses the Li automatic thresholding algorithm to segment
        the images."""
        # We crop the image because of some weird fringes.
        image = image[2:62,2:62]
        
        # We threshold the image.
        thresh = filters.threshold_li(image)
        img_binary = image > thresh
        
        # We keep only the largest object.
        labels = ndi.label(img_binary)[0]
        props = measure.regionprops(labels)
        areas = []
        for prop in props:
            areas.append((prop.area, prop.label))
        area_max = np.max(areas)
        for i in areas:
            if i[0] == area_max:
                target_label = i[1]
        labels_cleaned = np.zeros_like(image)
        labels_cleaned[labels == target_label] = 1
        
        return labels_cleaned * 1
    

#################### Processing of BIN files ##################################

# Reads the FLIM image.
img = IO_FLIM('D:/Microscope users/Sergio Lopez/Python/AI Capstone/Images/Pollen_1.bin')

# Gets the fast FLIM image.
flim_img = img.collection[0].get_FLIM(thresh=10)

# Gets the intensity image.
intensity_img = img.collection[0].get_int()

# Gets the decay of the whole image.
img_decay = img.collection[0].decay

# Gets the estimated IRF of the image (non-machine learning approach).
img_IRF = img.collection[0].IRF

# Estimates the IRF using a neural network.
img_IRF_NN = FLIM_methods.estimate_IRF_NN(img_decay[1])

# We plot the IRF extracted using the neural network.
x = np.arange(0.04848485,49.6,0.09696969)
plt.plot(x,img_IRF_NN)
plt.xlabel('Time / ns')
plt.ylabel('Intensity / a.u.')
plt.tight_layout()
plt.savefig('Extracted IRF.png')
plt.show()

# Gets the phasor plot.
phasor = img.collection[0].get_phasor()

# Displays the lifetime distribution.
FLIM_methods.display_lifetime_distribution(flim_img, savename='Lifetime distribution.png')

# Displays the FLIM image.
FLIM_methods.display_FLIM(intensity_img,flim_img, scalebar=True, lifetime_bar_lims=(1,3), resolution = img.collection[0].dictionary['PixResol'], save_file_name = 'FLIM.png')

# Displays the phasor plot.
frequency_laser = img.collection[0].dictionary['Frequency'] # This gets the laser frequency.
lifetimes = FLIM_methods.display_phasor(phasor,frequency=frequency_laser, clusters=1, savename='phasor.png')

# Fits the decay using the inferred IRF.
results = FLIM_methods.fit_decay(img_IRF,img_decay,num_decay=4)

# Displays the results of the fit.
FLIM_methods.display_fit(img_IRF,img_decay,results,save='Fit')

# Prints the fit report.
print(results.fit_report())

# Extracts the lifetime information from the fit.
info = FLIM_methods.extract_lifetimes(results,n_components=4)

# Extracts the lifetime information (q & t) from the fit using a neural network and a power-like model.
values_NN = FLIM_methods.extract_lifetime_NN(img_decay[1])
print(f'\n\nThe values according to the NN are t = {values_NN[0][0]} ns and q = {values_NN[0][1]}.\n\n')

# We segment the image using a Random Forest model.
rf_segmented = FLIM_methods.rf_segmenter(intensity_img)

# We display the image segmented using the Random Forest model.
plt.imshow(rf_segmented); plt.grid(visible=False); plt.title('Random Forest'); plt.savefig('RF.png'); plt.show()

# We segment the image using K-means clustering.
k_means_segmented = FLIM_methods.k_means_segmentation(flim_img, 2)

# We display the image segmented using K-means clustering.
plt.imshow(k_means_segmented); plt.grid(visible=False); plt.title('K means clustering'); plt.savefig('Kmeans.png'); plt.show()

# We segment the image using the Li segmenter.
li_segmented = FLIM_methods.li_segmenter(intensity_img)

# We display the image segmented using the Li segmenter.
plt.imshow(li_segmented); plt.grid(visible=False); plt.title('Li'); plt.savefig('Li.png'); plt.show()