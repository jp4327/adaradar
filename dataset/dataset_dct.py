import sys
import os

from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image

import random
from utils.dct import snr, dct_based_compression


class RADIal(Dataset):

    def __init__(self, root_dir,
                 statistics=None,
                 encoder=None,
                 difficult=False,
                 comp_ratio=1,
                 BL=64,
                 quantize=False, 
                 qbit=8,
                 verify_quantize=False,
                 cr_random=False,
                 cr_min=1,
                 cr_max=40,
                 ):

        self.root_dir = root_dir
        self.statistics = statistics
        self.encoder = encoder
        
        self.labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
       
        # Keeps only easy samples
        if(difficult==False):
            ids_filters=[]
            ids = np.where( self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]


        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.labels[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())
        

        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))

        self.comp_ratio = comp_ratio
        self.BL=BL
        self.quantize=quantize
        self.qbit=qbit
        self.verify_quantize=verify_quantize
        self.cr_random=cr_random
        self.cr_min = float(cr_min)
        self.cr_max = float(cr_max)


    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):
        
        # Get the sample id
        sample_id = self.sample_keys[index] 

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 


        ######################
        #  Encode the labels #
        ######################
        out_label=[]
        if(self.encoder!=None):
            out_label = self.encoder(box_labels).copy()      

        # Read the Radar FFT data
        radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npy".format(sample_id))
        input = np.load(radar_name,allow_pickle=True)
        radar_FFT = np.concatenate([input.real,input.imag],axis=2)
        if(self.statistics is not None):
            for i in range(len(self.statistics['input_mean'])):
                radar_FFT[...,i] -= self.statistics['input_mean'][i]
                radar_FFT[...,i] /= self.statistics['input_std'][i]

        radar_FFT = np.expand_dims(radar_FFT, axis=0) # unsqueeze
        radar_FFT = np.transpose(radar_FFT, (0,3,1,2)) # change order

        # Apply DCT, thresholding, quantization, IDCT
        if self.cr_random:
            self.comp_ratio = random.uniform(self.cr_min, self.cr_max)
        radar_FFT_comp, dct_coef = dct_based_compression(radar_FFT, self.comp_ratio, self.BL, self.quantize, self.qbit, self.verify_quantize)

        nonzero_count = np.count_nonzero(dct_coef) # count non-zero coefficients
        total_count = radar_FFT.size
        compression_ratio = total_count / nonzero_count if nonzero_count > 0 else float('inf')

        block_snr = snr(radar_FFT, radar_FFT_comp)

        radar_FFT_comp = np.transpose(radar_FFT_comp, (0,2,3,1)) # change to original order
        radar_FFT_comp = np.squeeze(radar_FFT_comp, axis=0)      # squeeze
        radar_FFT = radar_FFT_comp

        # Read the segmentation map
        segmap_name = os.path.join(self.root_dir,'radar_Freespace',"freespace_{:06d}.png".format(sample_id))
        segmap = Image.open(segmap_name) # [512,900]
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = self.crop(segmap)
        # and we resize to half of its size
        segmap = np.asarray(self.resize(segmap))==255

        # Read the camera image
        img_name = os.path.join(self.root_dir,'camera',"image_{:06d}.jpg".format(sample_id))
        image = np.asarray(Image.open(img_name))

        return radar_FFT, segmap,out_label,box_labels,image