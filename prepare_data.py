#!/usr/bin/env python3
import argparse
import sys
import os


def MXnet_record_to_folder(dataset_dir, save_dir=None):
    import os
    import numpy as np
    import mxnet as mx
    from tqdm import tqdm

    if save_dir is None:
        save_dir = (dataset_dir[:-1] if dataset_dir.endswith("/") else dataset_dir) + "_112x112_folders"
    idx_path = os.path.join(dataset_dir, "train.idx")
    bin_path = os.path.join(dataset_dir, "train.rec")

    print("save_dir = %s, idx_path = %s, bin_path = %s" % (save_dir, idx_path, bin_path))
    if os.path.exists(save_dir):
        print("%s already exists." % save_dir)
        return

    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, "r")
    rec_header, _ = mx.recordio.unpack(imgrec.read_idx(0))

    for ii in tqdm(range(1, int(rec_header.label[0]))):
        img_info = imgrec.read_idx(ii)
        header, img = mx.recordio.unpack(img_info)
        # img_idx = str(int(np.sum(header.label)))
        img_idx = str(int(header.label if isinstance(header.label, float) else header.label[0]))
        img_save_dir = os.path.join(save_dir, img_idx)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        with open(os.path.join(img_save_dir, str(ii) + ".jpg"), "wb") as ff:
            ff.write(img)


def MXnet_bin_files_to_tf(test_bins, save_bins,limit=0):
    import io
    import pickle
    import tensorflow as tf
    from skimage.io import imread

    print("test_bins =", test_bins)
    for test_bin_file,save_bin_file in zip(test_bins,save_bins):
        with open(test_bin_file, "rb") as ff:
            bins, issame_list = pickle.load(ff, encoding="bytes")

        bb = [tf.image.encode_jpeg(imread(io.BytesIO(ii))).numpy() for ii in bins[: limit * 2] + bins[-limit * 2 :]]
        print("Saving to %s" % save_bin_file)
        with open(save_bin_file, "wb") as ff:
            pickle.dump([bb, issame_list[:limit] + issame_list[-limit:]], ff)


""" CUDA_VISIBLE_DEVICES='-1' ./prepare_data.py -D /datasets/faces_emore """
""" CUDA_VISIBLE_DEVICES='-1' ./prepare_data.py -D /datasets/faces_emore -T lfw.bin cfp_fp.bin agedb_30.bin """
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-D", "--dataset_dir", type=str, required=True, help="MXnet record dataset directory")
    parser.add_argument("-T", "--test_bins", nargs="*", type=None, help="Test bin files in dataset_dir be converted")
    parser.add_argument("-S", "--save_dir", default=None, help="Folder path for saving dataset images")

    args = parser.parse_known_args(sys.argv[1:])[0]
    if args.test_bins is not None and len(args.test_bins) != 0:
        test_bins = [os.path.join(args.dataset_dir, ii) for ii in args.test_bins]
        save_bins = [os.path.join(args.save_dir, ii) for ii in args.test_bins]
        MXnet_bin_files_to_tf(test_bins,save_bins)
    MXnet_record_to_folder(args.dataset_dir, args.save_dir)
