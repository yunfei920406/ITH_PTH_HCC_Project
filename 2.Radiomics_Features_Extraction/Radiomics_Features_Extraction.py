import glob
import os
from Dcm_Preprocess import *
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
import numpy as np
from radiomics import featureextractor


def normalize_array(arr):
    # Find the minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Normalize the array
    normalized_arr = (arr - min_val) / (max_val - min_val)

    return normalized_arr


def extract_radiomics_features(image_path, mask_path, img_normalized=True, output_path=None, seq_name=None,
                               mask_name=None):
    # Read the image and mask
    if img_normalized:
        image = sitk.GetImageFromArray(normalize_array(Load_DcmFile(image_path)))
        mask = sitk.GetImageFromArray(Load_DcmFile(mask_path))
    else:
        image = sitk.GetImageFromArray(Load_DcmFile(image_path))
        mask = sitk.GetImageFromArray(Load_DcmFile(mask_path))

    # Get numpy arrays
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    image_ = sitk.GetImageFromArray(image_array)

    # Get all unique label values
    unique_labels = np.unique(mask_array[mask_array != 0])

    # Create an empty dataframe
    columns = []  # Replace with actual feature names based on your use case
    df = pd.DataFrame(columns=columns)

    # Create Pyradiomics feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    dfs = []

    # Iterate through each label, extract features, and add to the dataframe
    for label in unique_labels:
        try:
            # Get the region of interest (ROI) for the current label
            region_of_interest = np.where(mask_array == label, 1, 0)

            roi = sitk.GetImageFromArray(region_of_interest)

            # Extract Pyradiomics features
            features = extractor.execute(image_, roi)

            # Add the extracted features to the dataframe with label information
            dfs.append(pd.Series(features, name=f"Label_{label}"))
        except:
            pass

    df = pd.concat(dfs, axis=1)

    df.insert(0, "Features", df.index)

    index_ = []
    for i in df.index:
        if "diagnostics" in i:
            index_.append(False)
        else:
            index_.append(True)
    df = df.loc[index_, :]

    df.Features = [mask_name + "_" + seq_name + "_" + i for i in list(df.Features)]

    # Save results to the specified path
    if output_path is not None:
        df.to_excel(output_path, index=False)
    else:
        return df
