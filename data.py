import os
import glob
import argparse
import random
import csv
from shutil import copy, rmtree
from collections import defaultdict

from config import config


def clear():
    rmtree(config.TEST_DIR, ignore_errors=True)
    rmtree(config.TRAIN_DIR, ignore_errors=True)
    rmtree(config.VALIDATION_DIR, ignore_errors=True)
    os.makedirs(config.DATA , exist_ok=True)
    os.makedirs(config.TEST_DIR, exist_ok=True)
    os.makedirs(config.TRAIN_DIR, exist_ok=True)
    os.makedirs(config.VALIDATION_DIR, exist_ok=True)


def all_files(prefix="free"):
    return glob.glob(os.path.join(config.RAW_DIR, "{}-*".format(prefix)))


def all_identifiers(files):
    out = dict()
    for f in files:
        identifier = os.path.basename(f).split("-")[1]
        files = out.get(identifier, [])
        files.append(f)
        out[identifier] = files
    return out


def all_features(valid_features=None):
    feature = dict()
    imaging = dict()
    category = dict()
    with open(config.FEATURES) as f:
        reader = csv.DictReader(f, fieldnames=["id", "feature", "imaging", "category"])
        for row in reader:
            imaging[row["id"]] = row["imaging"]
            category[row["id"]] = row["category"]
            if valid_features is not None and row["feature"] not in valid_features:
                continue
            feature[row["id"]] = row["feature"]
    return feature, imaging, category

def all_test_set():
    out = list()
    with open(config.TEST_SET) as f:
        reader = csv.DictReader(f, fieldnames=["id"])
        for row in reader:
            out.append(row['id'])
        return out

def describe(prefix="free"):
    files = all_identifiers(all_files(prefix))
    feat, imag, category = all_features(['benign', 'malignant'])

    # create directories
    uniq_features = set(feat.values())
    for f in uniq_features:
        os.makedirs(os.path.join(config.TRAIN_DIR, f), exist_ok=True)
        os.makedirs(os.path.join(config.VALIDATION_DIR, f), exist_ok=True)
        os.makedirs(os.path.join(config.TEST_DIR, f), exist_ok=True)

    identifiers = list(files.keys())
    identifiers = [i for i in identifiers if i in feat]

    imaging_count = defaultdict(lambda x: 0)
    category_count = defaultdict(lambda x: 0)

    for i in identifiers:
        imaging_count[imag[i]] += 1
        category_count[category[i]] += 1
    return imaging_count, category_count

def sort(validation_split=0.2, prefix="free"):
    files = all_identifiers(all_files(prefix))
    feat, _, _ = all_features(['benign', 'malignant'])

    # create directories
    uniq_features = set(feat.values())
    for f in uniq_features:
        os.makedirs(os.path.join(config.TRAIN_DIR, f), exist_ok=True)
        os.makedirs(os.path.join(config.VALIDATION_DIR, f), exist_ok=True)
        os.makedirs(os.path.join(config.TEST_DIR, f), exist_ok=True)

    identifiers = list(files.keys())
    identifiers = [i for i in identifiers if i in feat]

    # separate out test set
    test = []
    if config.TEST_SET:
        test_source = all_test_set()
        test = [i for i in identifiers if i in test_source]
        # rest of identifiers without test set
        identifiers = [i for i in identifiers if i not in test_source]

    filtered_feat = { i: feat[i] for i in identifiers }
    identifiers_by_feature = defaultdict(list)
    for f, feature in filtered_feat.items():
        identifiers_by_feature[feature].append(f)

    validation = []
    train = []
    for feature, fs in identifiers_by_feature.items():
        random.shuffle(fs)
        split = int(len(fs) * validation_split)
        validation.extend(fs[:split])
        train.extend(fs[split:])

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

