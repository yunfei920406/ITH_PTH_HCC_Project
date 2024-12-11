# -*- coding: utf-8 -*-

import glob
import os
import shutil
import numpy as np
from tqdm import tqdm
from Dcm_Preprocess import Load_DcmFile
import SimpleITK as sitk
from skimage.segmentation import slic

def custom_slic_segmentation(image, mask, num_seg=50, compactness=0.1, enforce_connectivity=False):
    # Use the optimal num_seg for segmentation
    try:
        segments = slic(image, n_segments=num_seg, compactness=compactness, mask=mask, channel_axis=None, enforce_connectivity=enforce_connectivity)
        segments = segments.astype(int)
        return segments
    except:
        print("error")
