import os
import sys
import pdb
import time
import uuid

import dicom
import dicom_numpy
import scipy as sp
import numpy as np
import pandas as pd

import nibabel as nib

import SimpleITK as sitk
from scipy import ndimage as nd
from sklearn.feature_extraction import image

dir_path = os.path.dirname(os.path.realpath(__file__))
train_dir = dir_path.rstrip("src") + "traindata"
test_dir = dir_path.rstrip("src") + "testdata"

def register_patient(df, data_dir, data_type = "Train"):
    dir_path = "/Users/vester/Desktop/"
    for pid in df.ProxID.unique():
        temp = df.loc[df.ProxID == pid]
        try:
            # Get ADC, BVAL Dirs
            names = temp.Name.values
            adc_dir = temp[[name.endswith("ADC0") for name in names]].DCMSerUID.unique()[0]
            bval_dir = temp[[name.endswith("BVAL0") for name in names]].DCMSerUID.unique()[0]
            dicom_find = list(temp[[name.endswith("BVAL0") for name in names]].ijk.unique())
            dicom_find = [[int(coord) for coord in f.split(" ")] for f in dicom_find]
        except:
            continue

        dicom_path = data_dir + "/".join(["/raw/dicom", pid])
        ktrans_path = data_dir + "/".join(["/raw/ktrans", pid, pid])

        dicom2nifti(data_dir, 'adc', dicom_path, adc_dir, pid)
        dicom2nifti(data_dir, 'bval', dicom_path, bval_dir, pid)

        img = load_ktrans(ktrans_path)
        nimg = ktrans2nifti(data_dir, img, 'ktrans', pid, transpose = True)

    return


def dicom2nifti(main_dir, img_type, dicom_path, img_dir, pid):
    """
    #dicom2nifti
    dicom2nifti(data_dir, 'adc', dicom_path, adc_dir, pid)
    dicom2nifti(data_dir, 'bval', dicom_path, bval_dir, pid)
    """
    out_dir = "{0}/nifti/{1}/{2}".format(main_dir, img_type, pid)

    subdir = os.listdir(dicom_path)[-1]
    path = "/".join([dicom_path, subdir, img_dir])
    os.system("rm -rf {0}".format(out_dir))
    os.system("mkdir {0}".format(out_dir))
    os.system("/Applications/MRIcron/dcm2nii -e N -p N -d N -g N -i Y -n Y -o {0} {1}".format(out_dir, path))

    return


def ktrans2nifti(path, img, img_type, pid, transpose = False):
    """

    """
    print(pid)
    out_dir = "{0}/nifti/{1}/{2}".format(path, img_type, pid)
    path = "/".join([out_dir, pid])
    os.system("rm -rf {0}".format(out_dir))
    os.system("mkdir {0}".format(out_dir))

    affine = np.zeros((4,4))
    affine[:,-1] = np.append(img['origin'],1)
    for j in range(3):
        affine[j,:-1] = img['direction'][j*3:3*(j+1)]* img['scale'][-(j+1)]

    if transpose:
        nii = nib.Nifti1Image(img['arr'].T, affine.T)
    else:
        nii = nib.Nifti1Image(img['arr'], affine)

    nib.save(nii, path+'.nii')
    return nii


def load_ktrans(path):
    """
    # z,y,x
    """
    ext = "-Ktrans.mhd"
    img = sitk.ReadImage(path + ext)
    data = extract_info_from_sitk_img(img)

    return data


def extract_info_from_sitk_img(img):
    arr = sitk.GetArrayFromImage(img)
    origin = np.array(list(reversed(img.GetOrigin())), dtype = np.float)
    scale = np.array(list(reversed(img.GetSpacing())), dtype = np.float)
    direction = np.array(list(img.GetDirection()), dtype = np.float)

    data = {
        "img" : img,
        "arr" : arr,
        "origin" : origin,
        "scale" : scale,
        "direction" : direction
    }

    return data

if __name__ == '__main__':
    df_train = pd.read_csv(train_dir + "/ProstateX2-DataInfo-Train/ProstateX-2-Images-Train.csv")
    df_test = pd.read_csv(test_dir + "/ProstateX2-DataInfo-Test/ProstateX-2-Images-Test.csv")

    register_patient(df_train, train_dir, data_type = "Train")
    register_patient(df_test, test_dir, data_type = "Test")
