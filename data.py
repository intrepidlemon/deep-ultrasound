
# Modified from IX's original data.py intended for ultrasound images
# Last edits by BC
# TODO: had to save csv as MS-DOS csv in Excel?
# Note that Keras is only compatible up to Python 3.6

import os
import glob
import argparse
import random
import csv
import pandas
import numpy as np
from shutil import copy, rmtree
from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator

from config import config

def load_single(directory=config.VALIDATION_DIR):
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False)
    return generator

def data(shuffle_train=True, train_dir=config.TRAIN_DIR, validation_dir=config.VALIDATION_DIR, test_dir=config.TEST_DIR):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary')

    if not shuffle_train:
        train_generator = test_datagen.flow_from_directory(
            train_dir,
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            batch_size=config.BATCH_SIZE,
            class_mode='binary',
            shuffle=False)

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    return train_generator, validation_generator, test_generator

def clear():
    rmtree(config.TEST_DIR, ignore_errors=True)
    rmtree(config.TRAIN_DIR, ignore_errors=True)
    rmtree(config.VALIDATION_DIR, ignore_errors=True)
    os.makedirs(config.DATA , exist_ok=True)
    os.makedirs(config.TEST_DIR, exist_ok=True)
    os.makedirs(config.TRAIN_DIR, exist_ok=True)
    os.makedirs(config.VALIDATION_DIR, exist_ok=True)


# Returns list of path names from the raw directory
# The path names will be of the form RAW_DIR/*
def all_files(raw=config.RAW_DIR):
    return glob.glob(os.path.join(raw, "*"))

# Returns a dict of { picture ID# : [picture pathname] }
# Input is a list of path names
# TODO: how does scope of "files" work inside for loop?
def all_identifiers(files):
    out = dict()
    for f in files:
        identifier = os.path.basename(f).split(".")[0] # identifier = unique picture ID#
        files = out.get(identifier, [])
        files.append(f)
        out[identifier] = files
    return out

# Returns three dicts: {id:imaging}, {id:category}, {id:feature} from feature.csv
# Example usage: all_features(['benign', 'malignant'])
def all_features(valid_features=None, features=config.FEATURES):
    feature = dict()
    imaging = dict()
    category = dict()
    with open(features) as f:
        reader = csv.DictReader(f, fieldnames=["id", "feature", "imaging", "category"])
        for row in reader:
            imaging[row["id"]] = row["imaging"]
            category[row["id"]] = row["category"]
            # If feature of file is not desired, skip it and leave empty
            if valid_features is not None and row["feature"] not in valid_features:
                continue
            feature[row["id"]] = row["feature"]
    return feature, imaging, category

# Returns list of picture id that belong in test
def all_test_set():
    out = list()
    with open(config.TEST_SET) as f:
        reader = csv.DictReader(f, fieldnames=["id"])
        for row in reader:
            out.append(row['id'])
        return out

def describe(prefix="free", raw=config.RAW_DIR, features=config.FEATURES):
    files = all_identifiers(all_files(prefix, raw))
    feat, imag, category = all_features(['benign', 'malignant'], features)

    identifiers = list(files.keys())
    identifiers = [i for i in identifiers if i in feat]

    identifier_imaging_count = defaultdict(lambda: 0)
    identifier_category_count = defaultdict(lambda: 0)
    identifier_feat_count = defaultdict(lambda: 0)
    files_imaging_count = defaultdict(lambda: 0)
    files_category_count = defaultdict(lambda: 0)
    files_feat_count = defaultdict(lambda: 0)
    identifier_list = list()

    for i in identifiers:
        identifier_imaging_count[imag[i]] += 1
        identifier_category_count[category[i]] += 1
        identifier_feat_count[feat[i]] += 1
        files_imaging_count[imag[i]] += len(files[i])
        files_category_count[category[i]] += len(files[i])
        files_feat_count[feat[i]] += len(files[i])
        identifier_list.append({
            "identifier": i,
            "files": len(files[i]),
            "category": category[i],
            "feat": feat[i],
            "imag": imag[i],
        })
    return {
        "identifier": (identifier_feat_count, identifier_imaging_count, identifier_category_count),
        "files": (files_feat_count, files_imaging_count, files_category_count),
        "dataframe": pandas.DataFrame(identifier_list).set_index("identifier"),
    }

def print_describe(prefix="free", raw=config.RAW_DIR, features=config.FEATURES):
    identifier, files, df = describe(prefix, raw, features).values()
    columns = ["feat", "imag"]
    for category in np.unique(df.category):
        print("category:", category)
        print("\tcount:", len(df[df.category==category]))
        print("\tnumber lesions:", df[df.category==category].files.sum())
        for column in columns:
            print("\tcolumn:", column)
            values, counts = np.unique(df[df.category==category][column], return_counts=True)
            print("\t\t", dict(zip(values, counts)))


# Sort files from raw into train, test, and validation
# validation_split is fraction of pics in validation
def sort(validation_split=0.2, prefix="free"):
    files = all_identifiers(all_files())          # dict of { id : [picture pathname] }
    feat, _, _ = all_features(['benign', 'malignant'])  # dict of { id : feature }

    # create feature directories within train, validation, and test
    uniq_features = set(feat.values())
    for f in uniq_features:
        os.makedirs(os.path.join(config.TRAIN_DIR, f), exist_ok=True)
        os.makedirs(os.path.join(config.VALIDATION_DIR, f), exist_ok=True)
        os.makedirs(os.path.join(config.TEST_DIR, f), exist_ok=True)

    identifiers = list(files.keys())
    identifiers = [i for i in identifiers if i in feat]

    # separate out test set from identifiers
    test = []
    if config.TEST_SET:
        test_source = all_test_set()                    # list of picture id in test_source
        test = [i for i in identifiers if i in test_source]
        # rest of identifiers without test set
        identifiers = [i for i in identifiers if i not in test_source]

    filtered_feat = { i: feat[i] for i in identifiers } # dict {id:feature}, excluding test
    identifiers_by_feature = defaultdict(list)          # dict { feature : [id1, id2, ...]}
    for f, feature in filtered_feat.items():
        identifiers_by_feature[feature].append(f)

    # split pics into validation and train sets
    validation = []
    train = []
    for feature, fs in identifiers_by_feature.items():
        random.shuffle(fs)
        split = int(len(fs) * validation_split)
        validation.extend(fs[:split])
        train.extend(fs[split:])


    # add identified files to the intended directory
    for i in test:
        if i in feat:
            for f in files[i]:
                copy(f, os.path.join(config.TEST_DIR, feat[i]))
    for i in train:
        if i in feat:
            for f in files[i]:
                copy(f, os.path.join(config.TRAIN_DIR, feat[i]))
    for i in validation:
        if i in feat:
            for f in files[i]:
                copy(f, os.path.join(config.VALIDATION_DIR, feat[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split',
        type=float,
        default=0.2,
        help='percent of samples to set aside for validation')
    parser.add_argument(
        '--prefix',
        type=str,
        help='prefix of images [fixed, free]')
    FLAGS, unparsed = parser.parse_known_args()
    clear()
    sort(FLAGS.split, FLAGS.prefix)

