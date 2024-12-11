import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
import os
import glob
import pydicom as dicom


def Load_DcmSerieseFromFolders(folder_path):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path)
    filenames = reader.GetGDCMSeriesFileNames(folder_path, series_ids[0])
    reader.SetFileNames(filenames)
    sitk_img = reader.Execute()
    imgArray = sitk.GetArrayFromImage(sitk_img)
    return imgArray, sitk_img


def Load_DcmFile(file_path):
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)


def Get_Lesion_Index(arr):
    slice_index = []
    for i in range(np.min(arr.shape)):
        if np.sum(np.sum(arr, axis=2), axis=1)[i] != 0:
            slice_index.append(i)
    return slice_index


def GetMaxSliceNum(arr):
    """
    Determine the maximum slice number of lesion.
    Input_arr should be (Slices, H, W).
    """
    if len(arr.shape) != 3:
        print("The Input_Array should be 3-D like.")

    slices_num = np.min(arr.shape)
    arr = arr.reshape([slices_num, -1])

    Max_Slice_Index = np.argmax(np.sum(arr, axis=1))
    return Max_Slice_Index


def Covert2Uint8(arr):
    """
    arr should be 2D-shape.
    """
    arr = arr.astype("float32")
    max_ = np.max(arr)
    min_ = np.min(arr)
    range_ = max_ - min_

    out_ = (arr - min_) / range_ * 255
    out_ = out_.astype("uint8")
    return out_


def Load_DcmSeriesFromFoldersWithPydicom(folder_path):
    dcm_path_ls = glob.glob(os.path.join(folder_path, "*.dcm"))
    out_ls = []
    out_arr = []
    for file in dcm_path_ls:
        ds_ = dicom.read_file(file)
        out_ls.append(ds_)
        out_arr.append(ds_.pixel_array)
    out_arr = np.stack(out_arr)
    return out_arr, out_ls


def Save3DArray2Dcm(arr, ds_ls, output_folder_path):
    for i in range(arr.shape[0]):
        arr = np.int16(arr)
        slice_num = i + 1
        ds_ = ds_ls[i]
        ds_.decompress()
        ds_.PixelData = arr[i].tobytes()
        ds_.save_as(os.path.join(output_folder_path, str(slice_num) + ".dcm"))


def Load_Multi_Nii_to_4dArray(path, index):
    """
    :param path: the file path of folder containing multiple series.
    :param index: e.g., ["b0", "b10", "b20"].
    :return: 4D array.
    """
    path_list = os.listdir(path)
    arr_list = []
    for i in index:
        for j in path_list:
            if i + ".nii" == j:
                arr_list.append(Load_DcmFile(os.path.join(path, j)))
    return np.transpose(np.array(arr_list), (1, 2, 3, 0))


def SaveNiiFromArray(arr, path):
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, path)
