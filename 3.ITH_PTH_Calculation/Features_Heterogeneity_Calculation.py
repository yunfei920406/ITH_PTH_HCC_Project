
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def approximate_complex_numbers(df):
    """
    Check if there are any complex numbers in the DataFrame, and approximate them as integers.

    Parameters:
    df (DataFrame): The DataFrame to check.

    Returns:
    DataFrame: The DataFrame with complex numbers approximated as integers.
    """

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            try:
                float(df.iloc[i, j])
                judger = True
            except:
                judger = False

            if judger:
                df.iloc[i, j] = float(df.iloc[i, j])
            else:
                df.iloc[i, j] = 0.0

    return df


def normalize_df(df):
    normalized_df = df.iloc[:, 1:].copy()

    normalized_df = approximate_complex_numbers(normalized_df)
    num_row = normalized_df.shape[0]
    num_col = normalized_df.shape[1]
    for i in range(num_row):
        for j in range(num_col):
            row = normalized_df.iloc[i, :]
            max_val = row.max()
            min_val = row.min()

            if max_val == min_val:
                normalized_df.iloc[i, j] = 0
            else:
                normalized_df.iloc[i, j] = (normalized_df.iloc[i, j] - min_val) / (max_val - min_val)
    return normalized_df

def cal_het_rad(df):
    features_names = list(df.iloc[:, 0])
    normalized_df = normalize_df(df)
    mean_ = np.mean(normalized_df, axis=1)

    sd_ = np.std(normalized_df, axis=1)
    try:
        het = sd_ / mean_
        het = np.nan_to_num(het, nan=0)
    except:
        het_ls = []
        for i, j in zip(sd_, mean_):
            try:
                het_ls.append(i / j)
            except:
                het_ls.append(0.0)
        het = het_ls
    df_ = pd.DataFrame(het).T
    df_.columns = features_names
    return df_

def cal_conven_rad(df):
    features_names = list(df.iloc[:, 0])
    rad = pd.DataFrame(np.array(df.iloc[:, 1])).T

    rad.columns = features_names

    return rad

def batch_cal_features(base_path_raw, conv_label="HCC", out_path_folder_name="Conv_Het_Features"):

    out_path_folder = os.path.join(os.path.dirname(base_path_raw), out_path_folder_name)
    if not os.path.exists(out_path_folder):
        os.mkdir(out_path_folder)

    ls_het_all_pat = []
    ls_conv_all_pat = []
    name_suc = []
    name_err = []

    for name in tqdm(os.listdir(base_path_raw)):
        try:
            ls_conv = []
            ls_het = []
            for file in os.listdir(os.path.join(base_path_raw, name)):
                if conv_label in file:
                    out = cal_conven_rad(pd.read_excel(os.path.join(base_path_raw, name, file)))
                    ls_conv.append(out)
                else:
                    out = cal_het_rad(pd.read_excel(os.path.join(base_path_raw, name, file)))
                    ls_het.append(out)
            het_all = pd.concat(ls_het, axis=1)
            conv_all = pd.concat(ls_conv, axis=1)

            ls_conv_all_pat.append(conv_all)
            ls_het_all_pat.append(het_all)

            name_suc.append(name)
        except:
            name_err.append(name)

    df_rad_conv = pd.concat(ls_conv_all_pat, axis=0)
    df_rad_het = pd.concat(ls_het_all_pat, axis=0)

    df_rad_conv.insert(0, "Name", name_suc)
    df_rad_het.insert(0, "Name", name_suc)

    df_rad_het.to_csv(os.path.join(out_path_folder, "df_rad_het.csv"), index=False)
    df_rad_conv.to_csv(os.path.join(out_path_folder, "df_rad_conv.csv"), index=False)

    return name_err