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
import SimpleITK as sitk

from visualizer import *
from scipy import ndimage as nd
from sklearn.feature_extraction import image


dir_path = os.path.dirname(os.path.realpath(__file__))
train_dir = dir_path.rstrip("src") + "traindata"
test_dir = dir_path.rstrip("src") + "testdata"
PATCH_SUM_TRESHOLD = 1e-8
PATCH_SHAPE = (32,32)
MAX_SAVE_COUNT = 1500
np.random.seed(1337)


def get_patches(df, data_dir, img_type = "ADC", data_type = "Train"):
    """
    Loop over df, grab ADC/ slices and save to one
    """
    print("Cleaning Output Directory")
    os.system("rm -rf %s" % (data_dir + "/patches/" + img_type ) )
    os.system("mkdir %s" % (data_dir+"/patches/" + img_type) )
    os.system("mkdir %s" % (data_dir+"/patches/" + img_type + "/POS") )
    os.system("mkdir %s" % (data_dir+"/patches/" + img_type + "/NEG") )

    hlen = 113
    header = img_type + " - " + data_type
    prl = (hlen//2-len(header)//2) - 1
    prr = hlen - prl - len(header) - 2
    print("-"*hlen)
    print("#" + " "*prl + header + " "*prr + "#")
    print("-"*hlen)

    print("{0:<15s}\t{1:>5s}\t{2:>10s}\t{3:>10s}\t{4:>6s}\t{5:>6s}\t{6:>6s}\t{7:>6s}"\
        .format("PID", "#Find", "#Patch", "#Positive", "#Negative", "Load time", "Extract Time", "Save Time"))
    overall = time.time()
    for pid in df.ProxID.unique():
        load_start = time.time()
        print("{0:<15s}"
            .format("Loading"), end = "\r", flush = True)

        temp = df.loc[df.ProxID == pid]
        try:
            # Get ADC, BVAL Dirs
            names = temp.Name.values
            adc_dir = temp[[name.endswith("ADC0") for name in names]].DCMSerUID.values[0]
            bval_dir = temp[[name.endswith("BVAL0") for name in names]].DCMSerUID.values[0]
            dicom_find = list(temp[[name.endswith("BVAL0") for name in names]].ijk.unique())
            dicom_find = [[int(coord) for coord in f.split(" ")] for f in dicom_find]

        except:
            continue

        dicom_path = data_dir + "/".join(["/raw/dicom", pid])
        # ADC Processing
        if img_type == "ADC":
            imgs, findings = load_dicom(dicom_path, adc_dir, dicom_find)
        elif img_type == "BVAL":
            imgs, findings = load_dicom(dicom_path, bval_dir, dicom_find)
        elif img_type == "KTRANS":
            ktrans_path = data_dir + "/".join(["/raw/ktrans", pid, pid])
            imgs = load_ktrans(ktrans_path)
            findings = []
        else:
            print("image type not defined! ABORT")
            return -1

        ltime = time.time() - load_start

        ext_start = time.time()
        fcount = len(dicom_find)
        print("{0:<15s}\t{1:>5d}".format("Extracting", fcount), end = "\r", flush = True)
        if img_type == "ADC" or img_type == "BVAL":
            # Patches without, Pathces with findings
            pwof, pwf = extract_patches(imgs, PATCH_SHAPE, findings)
        elif img_type == "KTRANS":
            print("KTRANS PROCESSING NOT YET IMPLEMENTED! ABORT")
            return -1
        etime = time.time() - ext_start


        save_start = time.time()

        # Printing
        p = len(pwf)
        n = len(pwof)

        print("{0:<15s}\t{1:>5d}\t{2:>10s}\t{3:>10s}\t   {4:>6s}\t   {5:>6s}\t      {6:>6s}"\
            .format("Saving", fcount, str(p+n), str(p), str(n), str(ltime)[:6], str(etime)[:6]), end = "\r", flush = True)

        pwf = np.array(pwf)
        pwof = np.array(pwof)
        n = save_patches(pid, fcount, data_dir, img_type, pwf, pwof)

        stime = time.time() - save_start
        print("{0:15s}\t{1:>5d}\t{2:>10s}\t{3:>10s}\t   {4:>6s}\t   {5:>6s}\t      {6:>6s}\t   {7:>6s}"\
            .format(pid, fcount,  str(n+n), str(n), str(n), str(ltime)[:6], str(etime)[:6], str(stime)[:6]))

        #break

    otime = time.time() - overall

    print("-"*hlen)
    header = "time taken:" + "{0:4.5f}".format(otime)
    prl = (hlen//2-len(header)//2) - 1
    prr = hlen - prl - len(header) - 2
    print("#" + " "*prl + header + " "*prr + "#")
    print("-"*hlen)

    return 0


def save_patches(pid, fcount, data_dir, img_type, pwf, pwof):
    dname = data_dir + "/patches/" + img_type + "/"
    fname = "POS/POS_" + img_type+ "_" + pid[-4:]
    #pdb.set_trace()
    #np.save(dname + fname, pwf)
    idxs = np.random.randint(0, pwf.shape[0], size = MAX_SAVE_COUNT * fcount)
    pwf = pwf[idxs]
    for j in range(pwf.shape[0]):
        path = dname + fname + "_" + str(uuid.uuid4())[-12:] + ".png"
        sp.misc.imsave(path,pwf[j])

    fname = fname.replace("POS", "NEG")
    idxs = np.random.randint(0, pwof.shape[0], size = MAX_SAVE_COUNT * fcount)
    pwof = pwof[idxs]
    for j in range(pwof.shape[0]):
        path = dname + fname + "_" + str(uuid.uuid4())[-12:] + ".png"
        sp.misc.imsave(path,pwof[j])
    #idxs = np.random.randint(0, pwof.shape[0], size = pwf.shape[0])
    #np.save(dname + fname, pwof[idxs])
    #pdb.set_trace()
    return len(idxs)


def load_dicom(path, dicom_dir, ijk_finding = []):
    subdir = os.listdir(path)[-1]
    dicom_dir = "/".join([path, subdir, dicom_dir])
    xscale, yscale, zscale = [], [], []
    slices = []
    for files in os.listdir(dicom_dir):
        if not files.endswith("dcm"):
            continue
        filepath = dicom_dir + "/" + files
        slices.append(filepath)

    ds = [dicom.read_file(fp) for fp in slices]
    arr, trans = dicom_numpy.combine_slices(ds)

    # Get Info
    scale = [*ds[0].PixelSpacing, ds[0].SliceThickness]
    scale = np.array(scale, dtype = np.int)

    orient = ds[0].ImageOrientationPatient
    pos = ds[0].ImagePositionPatient

    # Rescale
    arr = rescale_arr(arr, scale)
    ijk_finding = [(np.array(finding) * scale).tolist() for finding in ijk_finding]
    limits = [(0,l) for l in arr.shape]

    findings = []
    for finding in ijk_finding:
        findings.extend(voxel_to_cube(finding, limits, scale))

    findings = np.array(findings)
    findings = np.unique(findings, axis = 0)

    return arr, findings


def voxel_to_cube(center, limits, scale):
    counter = 0
    coordinates = []

    for j in range(center[0] - scale[0] // 2, center[0] + (scale[0] // 2) + 1 ,1):
        for k in range(center[1] - scale[1] // 2, center[1] + (scale[1] // 2) + 1 ,1):
            for l in range(center[2] - scale[2] // 2, center[2] + (scale[2] // 2) + 1 ,1):
                counter += 1
                if check_limits((j,k,l), limits):
                    coordinates.append((j,k,l))
                    #print((j,k,l))
    #print(counter, len(coordinates))
    return coordinates


def check_limits(coordinate, limit):
    j,l,k = coordinate
    res = 0
    if (j > limit[0][0] and j < limit[0][1] - 1):
        res += 1
    if (l > limit[1][0] and l < limit[1][1] - 1):
        res += 1
    if (k > limit[2][0] and k < limit[2][1] - 1):
        res += 1

    return res == 3


def rescale_arr(arr, scale):
    arr = nd.interpolation.zoom(arr.astype(np.float), zoom = scale, order = 1)
    arr /= arr.sum()
    arr[np.abs(arr) < 1e-16] = 0
    return arr


def orient_dicom(arr, xscale, yscale, zscale):
    print("What the f")
    return arr


def load_ktrans(path):
    """
    # z,y,x
    #arr = io.imread(path + ext, plugin = 'simpleitk')
    #arr = arr / arr.sum()
    """
    ext = "-Ktrans.mhd"
    img = sitk.ReadImage(path + ext)

    # Order z,y,x
    arr = sitk.GetArrayFromImage(img)
    origin = np.array(list(reversed(img.GetOrigin())))
    scale = np.array(list(reversed(img.GetSpacing())))
    arr = rescale_arr(arr, scale)

    return arr


def extract_patches(arr3d, patch_shape, findings):
    """
    ADC, BVAL input
    """
    res_pwof = []
    res_pwf = []

    js = np.unique(findings[:,2])

    for j in range(arr3d.shape[2]):
        arr = arr3d[:,:,j]
        if j in js:
            sub_findings = findings[np.where(findings[:,2] == j)]
            pwof, pwf = extract_patches_loop(arr, patch_shape, sub_findings)
            res_pwf.extend(pwf)
            res_pwof.extend(pwof)
        else:
            pwof = image.extract_patches_2d(arr, patch_shape, max_patches = 5000)
            pwof = filter_empty_patches(pwof)
            res_pwof.extend(pwof)
            pass

    return res_pwof, res_pwf


def extract_patches_loop(arr, patch_shape, sub_findings):
    """
    Extract Pathces with potential findings.
    """
    arr_row, arr_col = arr.shape
    patch_row, patch_col = patch_shape

    # Extract findings and patches
    findings = sub_findings[:,:2]
    patches = image.extract_patches_2d(arr, patch_shape)

    # Patches per row/column
    ppc =  arr_row - patch_row + 1
    ppr =  arr_col - patch_col + 1

    # Mapping between indexing
    i2c = lambda idx: (idx // ppr, idx % ppr)
    c2i = lambda xs:  xs[0] * ppr + xs[1]
    iden = lambda j: c2i(i2c(j))
    funcx = lambda fx: np.arange(patch_row) + (fx - patch_row + 1)
    funcy = lambda fy: np.arange(patch_col) + (fy - patch_col + 1)

    # Extract patches with findings
    idx = findings_2_idx(findings, c2i, funcx, funcy)
    mask = ~np.ones(patches.shape[0], dtype = bool)
    mask[idx] = True

    patches_with_findings = filter_empty_patches(patches[mask])
    patches_without_findings = filter_empty_patches(patches[~mask])

    return patches_without_findings, patches_with_findings


def findings_2_idx(findings, corner_2_idx, funcx, funcy):
    """
    Map findings coodinates to patch index
    """
    idx = []
    for finding in findings:
        x, y = finding
        mesh = np.array(np.meshgrid(funcx(x), funcy(y))).swapaxes(1,2).reshape(2,-1).T
        idx.extend([corner_2_idx(c) for c in mesh])

    return np.unique(idx)


def filter_empty_patches(patches):

    patch_sums = np.sum(patches, axis = (1,2))
    patches = patches[patch_sums > PATCH_SUM_TRESHOLD]

    return patches


if __name__ == '__main__':
    df_train = pd.read_csv(train_dir + "/ProstateX2-DataInfo-Train/ProstateX-2-Images-Train.csv")
    df_test = pd.read_csv(test_dir + "/ProstateX2-DataInfo-Test/ProstateX-2-Images-Test.csv")

    get_patches(df_train, train_dir, img_type = "ADC", data_type = "Train")
    get_patches(df_test, test_dir, img_type = "ADC", data_type = "Test")

    #get_patches(df_train, train_dir, img_type = "BVAL", data_type = "Train")
    #get_patches(df_test, test_dir, img_type = "BVAL", data_type = "Test")
