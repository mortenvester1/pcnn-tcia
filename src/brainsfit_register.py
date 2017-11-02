import os
import re
import sys
import pdb
import time
import subprocess
import scipy as sp
import numpy as np
import pandas as pd
import nibabel as nib

from scipy import ndimage as nd
from util import *

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = DIR_PATH.rstrip("src") + "traindata"
TEST_DIR = DIR_PATH.rstrip("src") + "testdata"

TYPES = ["adc", "bval", "ktrans"]
FIXED = "adc"

def register_patients(df, data_dir, fixed_type, mov_types):
    OUT_WIDTH = 81
    data_dir += "/nifti/"

    print_border(OUT_WIDTH)
    print_header("BRAINSFit Registration", OUT_WIDTH)
    print_header("Fixed: {0:s}, Moving: {1:s}"\
                 .format(fixed_type, ",".join(mov_types)), OUT_WIDTH)
    print_header("dir: {0:s}".format(data_dir), OUT_WIDTH)
    print_border(OUT_WIDTH)

    start = time.time()


    print("{0:<14s}\t\t{1:>9s}\t\t{2:>9s}\t\t{3:>9s}"\
          .format("pid", "#success", "iterations", "regtime"))
    errors = []
    for pid in df.ProxID.unique():
        print("{0:<14s}".format("BRAINSFit"), end = '\r', flush = True)

        reg_start = time.time()
        fix_path, mov_paths, out_paths = get_paths(pid, data_dir, fixed_type, mov_types)
        suc, err = brainsfit_registration(fix_path, mov_paths, out_paths)
        reg_time = time.time() - reg_start

        if len(suc) == len(mov_types):
            iters = suc[0][0], suc[1][0]
        else:
            iters = (0,0)
            errors.append(err)

        # rescale
        print("{0:<14s}\t\t{1:>9d}\t\t {2:>4d},{3:>4d}\t\t{4:>9.5f}".format\
             (pid, len(suc), iters[0], iters[1], reg_time))

        """
        scale = []
        scale_start = time.time()
        rescale_niis(out_paths, scale)
        scale_time = time.time() - scale_start
        print("{0:<14s}\t{1:>9d}\t{2:>4d},{3:>4d}\t{4:>5.4f}\t{5:>5.4f}".format\
             (pid, len(suc), iters[0], iters[1], reg_time, scale_time),
              end = "\r")
        """

    elapsed = time.time() - start

    print_border(OUT_WIDTH)
    print_header("Elapsed time: {0:f} Seconds".format(elapsed), OUT_WIDTH)
    if errors:
        print_header("Found errors in {0:d} files".format(len(errors)), OUT_WIDTH)
        with open(data_dir + "notes.txt", 'w') as notes:
            out = "\n ".join([str(err) for err in errors])
            notes.write(str(out))

    print_border(OUT_WIDTH)

    return errors


def brainsfit_registration(fix_path, mov_paths, out_paths):
    """
    https://www.slicer.org/wiki/Slicer3:Registration
    https://www.slicer.org/wiki/Modules:BRAINSFit
    """
    template = """BRAINSFit --fixedVolume {0:s}
                 --movingVolume {1:s}
                 --outputVolume {2:s}
                 --transformType Affine
                 --initializeTransformMode useGeometryAlign
                 --numberOfSamples 200000"""

    errors = []
    success = []
    for j, mov_path in enumerate(mov_paths):
        try:
            start = time.time()

            BRAINSFit_call = template.format(fix_path, mov_path, out_paths[j+1])
            BRAINSFit_call = re.sub("[ ]+"," ", BRAINSFit_call).replace("\n","")
            output = subprocess.check_output([BRAINSFit_call], shell=True, executable='/bin/bash')
            elapsed = time.time() - start
            count = parse_brainsfit_output(output)

            if count > 0:
                success.append((count, elapsed))
            else:
                errors.append(mov_path)
        except:
            errors.append(mov_path)

    cp_fixed = "cp {0:s} {1:s}".format(fix_path, out_paths[0])
    output = subprocess.check_output(cp_fixed, shell=True, executable='/bin/bash')

    return success, errors


def parse_brainsfit_output(output):
    try:
        res = str(output).split("\\n")[-2]
    except:
        return -1

    if "Convergence" in res:
        count = int(re.search("(?<=iteration )[0-9]+",res).group(0))

    return count


def get_paths(pid, data_dir, fixed_type, mov_types):
    in_template = data_dir + "{0:s}/{1:s}/{1:s}.nii"
    fixed_path = in_template.format(fixed_type, pid)
    moving_paths = [in_template.format(mov, pid) for mov in mov_types]

    out_template = data_dir + "registered/{0:s}{1:s}.nii"
    out_paths = [out_template.format(pid, fixed_type)] +\
                [out_template.format(pid, mov) for mov in mov_types]

    return fixed_path, moving_paths, out_paths


def rescale_niis(paths, scale):
    affine = np.eye(4)
    for path in paths:
        img = nib.load(path)
        arr = rescale_arr(arr, scale)
        nib.save(path, nib.Nifti1Image(arr, affine) )

    return


def rescale_arr(arr, scale):
    arr = nd.interpolation.zoom(arr.astype(np.float), zoom = scale, order = 1)
    arr /= arr.sum()
    arr[np.abs(arr) < 1e-16] = 0
    return arr


def normalize_arr(arr, norm_type = "std"):
    if norm_type == "minmax":
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        arr_min_max = (arr - arr_min)/(arr_max - arr_min)
        return arr_min_max
    else:
        arr = np.divide(arr,arr.sum())
    return arr


if __name__ == '__main__':
    df_train = pd.read_csv(TRAIN_DIR + "/ProstateX2-DataInfo-Train/ProstateX-2-Images-Train.csv")
    df_test = pd.read_csv(TEST_DIR + "/ProstateX2-DataInfo-Test/ProstateX-2-Images-Test.csv")

    register_patients(df_train, TRAIN_DIR, 'adc', ['bval','ktrans'] )
    register_patients(df_test, TEST_DIR, 'adc', ['bval','ktrans'] )
