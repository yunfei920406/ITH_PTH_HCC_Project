from Dcm_Preprocess import *
from scipy.ndimage import binary_dilation
import numpy as np
import cv2
from tqdm import tqdm
import SimpleITK as sitk

def dilate_mask_3d(mask_array, size=1, return_peri=True):
    """
    Perform 3D mask dilation using 3D convolution.

    Parameters:
    - mask_array: 3D array, where 0 represents background and 1 represents the mask
    - size: dilation size, i.e., the number of iterations, default is 1
    - voxel_value: the new value for the dilated areas, default is 2

    Returns:
    - Dilated 3D array
    """
    dilated_array = binary_dilation(mask_array, iterations=size).astype(np.uint8)

    if return_peri:
        return dilated_array, dilated_array - mask_array
    else:
        return dilated_array
