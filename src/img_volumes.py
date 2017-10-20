import os
import pdb
import sys
import time
import uuid

import h5py
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from skimage.util import view_as_windows
from scipy import ndimage as nd
from sklearn.feature_extraction import image

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = SRC_DIR.rstrip("src") + "traindata"
TEST_DIR = SRC_DIR.rstrip("src") + "testdata"
VOL_SUM_TRESHOLD = 1e-8
VOL_SHAPE = (32,32,12)
VOL_STRIDE = (4,4,2)
np.random.seed(1337)
HLEN = 113

IMG_TYPES = ['adc','bval','ktrans']

def get_volumes(df, data_dir, img_types = [], data_type = "Train"):
    """
    Loop over df, grab ADC/ slices and save to one
    """
    if not img_types:
        print("ABORT: No images types specified")
        return

    print("Cleaning Output Directory")
    os.system("rm -rf %s" % (data_dir + "/volumes/" ) )
    os.system("mkdir %s" % (data_dir+"/volumes/") )
    #os.system("mkdir %s" % (data_dir+"/volumes/POS") )
    #os.system("mkdir %s" % (data_dir+"/volumes/NEG") )

    print_border()
    header = ",".join(img_types) + " - " + data_type
    print_header(header)
    print_border()

    h5storage = h5py.File(data_dir+"/volumes/"+data_type.lower()+".h5",'w')

    print("{0:<15s}\t{1:>5s}\t{2:>10s}\t{3:>10s}\t{4:>6s}\t{5:>6s}\t{6:>6s}\t{7:>6s}"\
        .format("PID", "#Find", "#Volume", "#Positive", "#Negative", "Load time", "Extract Time", "Save Time"))
    overall = time.time()
    for pid in df.ProxID.unique():
        load_start = time.time()
        print("{0:<15s}"
            .format("Loading"), end = "\r", flush = True)

        temp = df.loc[df.ProxID == pid]
        try:
            # Get ADC, BVAL Dirs
            names = temp.Name.values
            findings = list(temp[[name.endswith("BVAL0") for name in names]].ijk.unique())
            findings = [[int(coord) for coord in f.split(" ")] for f in findings]
            fc = len(findings)
            scale = list(temp[[name.endswith("BVAL0") for name in names]].VoxelSpacing.unique())[0]
            scale = np.array(scale.split(','), dtype = np.int)
        except:
            print("{0:<15s}\t{1:>5s}".format(pid,"SKIP"))
            continue

        ##### LOADING NIFTI
        l_start = time.time()
        imgs = load_nifti(data_dir, pid, img_types)
        ltime = time.time() - l_start
        if imgs is None:
            print("{0:<15s}\t{1:>5s}".format(pid,"SKIP"))
            continue

        ##### EXTRACTING VOLUMES
        e_start = time.time()
        print("{0:<15s}\t{1:>5d}".format("Extracting", fc), end = "\r", flush = True)
        pos, neg = extract_volumes(imgs, scale, findings)
        etime = time.time() - e_start


        ##### SAVING VOLUMES
        s_start = time.time()
        p = len(pos[0])
        n = len(neg[0])
        print("{0:<15s}\t{1:>5d}\t{2:>10s}\t{3:>10s}\t   {4:>6s}\t   {5:>6s}\t      {6:>6s}"\
                .format("Saving", fc, str(p+n), str(p), str(n), str(ltime)[:6], str(etime)[:6]), end = "\r", flush = True)
        save_vols(data_dir + '/volumes/', pid, pos, neg, h5storage)
        stime = time.time() - s_start

        ##### Print stats
        print("{0:15s}\t{1:>5d}\t{2:>10s}\t{3:>10s}\t   {4:>6s}\t   {5:>6s}\t      {6:>6s}\t   {7:>6s}"\
                .format(pid, fc,  str(n+n), str(n), str(n), str(ltime)[:6], str(etime)[:6], str(stime)[:6]))
        #break

    otime = time.time() - overall
    print_border()
    header = "time taken:" + "{0:4.5f}".format(otime)
    print_header(header)
    print_border()

    h5storage.close()

    return


def extract_volumes(imgs, scale, findings):
    arrs = [rescale_arr(img.get_data(), scale) for img in imgs]
    findings = rescale_findings(findings, scale, arrs[0].shape )
    pos_mask, neg_mask = finding_2_vol_idx(arrs[0].shape, findings)

    pos, neg = [], []
    #return pos, neg
    for arr in arrs:
        vols = view_as_windows(arr, VOL_SHAPE, VOL_STRIDE )
        vols = vols.reshape([-1]+list(VOL_SHAPE))

        pos.append(vols[pos_mask])
        neg.append(vols[neg_mask])

    return pos, neg


def split_pos_neg(mask, vols):
    sample_size = len(mask)
    pos = vols[mask]
    neg = vols[~mask]

    return pos, neg


def finding_2_vol_idx(shape, findings):
    """
    Create array of zeros and mark finding.
    generate volumes and test
    """
    arr = np.zeros(shape)
    arr[findings[:,0],findings[:,1],findings[:,2]] = 1
    vols = view_as_windows(arr, VOL_SHAPE, VOL_STRIDE )
    vols = vols.reshape([-1]+list(VOL_SHAPE))
    sums = np.sum(vols, axis = (1,2,3))
    pos_mask = sums > 0
    neg_idx = np.where(pos_mask == False)[0]
    neg_idx = np.random.choice(neg_idx, pos_mask.sum(), replace = False)

    neg_mask = np.zeros(len(pos_mask),dtype=bool)
    neg_mask[neg_idx] = True

    assert (neg_mask * pos_mask).sum() == 0

    return pos_mask, neg_mask


def rescale_arr(arr, scale):
    arr = nd.interpolation.zoom(arr.astype(np.float), zoom = scale, order = 1)
    arr /= arr.sum()
    arr[np.abs(arr) < 1e-16] = 0
    return arr


def rescale_findings(ijk_findings, scale, arr_shape):
    ijk_finding = [(np.array(finding) * scale).astype(np.int).tolist() for finding in ijk_findings]
    limits = [(0,l) for l in arr_shape]

    findings = []
    for finding in ijk_finding:
        findings.extend(voxel_to_cube(finding, limits, scale))
    findings = np.array(findings)
    findings = np.unique(findings, axis = 0)

    return findings


def voxel_to_cube(center, limits, scale):
    counter = 0
    coordinates = []

    for j in range(center[0] - scale[0] // 2, center[0] + (scale[0] // 2) + 1 ,1):
        for k in range(center[1] - scale[1] // 2, center[1] + (scale[1] // 2) + 1 ,1):
            for l in range(center[2] - scale[2] // 2, center[2] + (scale[2] // 2) + 1 ,1):
                counter += 1
                if check_limits((j,k,l), limits):
                    coordinates.append((j,k,l))
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


def load_nifti(path, pid, img_types):
    ext = '.nii'
    paths = ["/".join([path, 'nifti', 'registered', pid + img_t + ext]) for img_t in img_types]
    if not all(os.path.isfile(fp) for fp in paths):
        return None

    imgs = [nib.load(fp) for fp in paths]
    return imgs


def save_vols(path, pid, pos, neg, h5storage):
    #ext = ".npz"
    #ext = ".nii.gz"
    #pos_path = [path+"/POS/" + it[0]+pid[-4:] for it in IMG_TYPES]
    #neg_path = [path+"/NEG/" + it[0]+pid[-4:] for it in IMG_TYPES]
    #uuids = gen_ids(len(pos[0]))

    #affine = np.diag([1,1,1,1])
    for j, p in enumerate(pos):
        for k, vol in enumerate(p):
            h5storage.create_dataset("POS/" + IMG_TYPES[j] + "/" + pid[-4:] + "-{0:05d}".format(k), data = vol.astype(np.float64))
            #vol.astype(np.float16).tofile(pos_path[j] + "-{0:05d}".format(k))
            #sp.sparse.save_npz(pos_path[j] + "-{0:05d}".format(k) + ext,sp.sparse.spmatrix(vol))
            #np.save(pos_path[j] + "-{0:05d}".format(k) + ext, vol)
            #nii = nib.Nifti1Image(vol, affine)
            #nib.save(nii, pos_path[j] + "-{0:05d}".format(k) + ext)
            #pdb.set_trace()

    for j, n in enumerate(neg):
        for k, vol in enumerate(n):
            h5storage.create_dataset("NEG/" + IMG_TYPES[j] + "/" + pid[-4:] + "-{0:05d}".format(k), data = vol.astype(np.float64))
            #vol.astype(np.float16).tofile(neg_path[j] + uuids[k])
            #sp.sparse.save_npz(neg_path[j] + uuids[k] + ext,sp.sparse.spmatrix(vol))
            #np.save(neg_path[j] + uuids[k] + ext, vol)
            #nii = nib.Nifti1Image(vol, affine)
            #nib.save(nii, neg_path[j] + uuids[k] + ext)
    return




def gen_ids(n):
    uuids = [str(uuid.uuid4())[-12:] for k in range(n)]
    return uuids


def print_header(header):
    prl = (HLEN//2-len(header)//2) - 1
    prr = HLEN - prl - len(header) - 2
    print("#" + " "*prl + header + " "*prr + "#")
    return


def print_border():
    print("-"*HLEN)
    return


if __name__ == '__main__':
    df_train = pd.read_csv(TRAIN_DIR + "/ProstateX2-DataInfo-Train/ProstateX-2-Images-Train.csv")
    df_test = pd.read_csv(TEST_DIR + "/ProstateX2-DataInfo-Test/ProstateX-2-Images-Test.csv")

    get_volumes(df_train, TRAIN_DIR, img_types = IMG_TYPES, data_type = "Train")
    get_volumes(df_test, TEST_DIR, img_types = IMG_TYPES, data_type = "Test")
