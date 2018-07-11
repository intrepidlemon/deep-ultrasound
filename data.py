import os
import glob
import argparse
import random
import csv
from shutil import copy, rmtree
from collections import defaultdict

from config import config

def clear():
    rmtree(config.TRAIN_DIR, ignore_errors=True)
    rmtree(config.VALIDATION_DIR, ignore_errors=True)
    os.makedirs(config.DATA , exist_ok=True)
    os.makedirs(config.TRAIN_DIR, exist_ok=True)
    os.makedirs(config.VALIDATION_DIR, exist_ok=True)


def all_files(prefix="free"):
    return glob.glob(os.path.join(config.RAW_DIR, "{}-*".format(prefix)))


def all_identifiers(files):
    out = dict()
    for f in files:
        out[os.path.basename(f).split("-")[1]] = f
    return out


def all_features(valid_features=None):
    out = dict()
    with open(config.FEATURES) as f:
        reader = csv.DictReader(f, fieldnames=["id", "feature"])
        for row in reader:
            if valid_features is not None and row["feature"] not in valid_features:
                continue
            out[row["id"]] = row["feature"]
    return out


def sort(percent_split=0.4):
    files = all_identifiers(all_files())
    feat = all_features(['benign', 'malignant'])

    identifiers = list(files.keys())
    identifiers = [i for i in identifiers if i in feat]
    filtered_feat = { i: feat[i] for i in identifiers }

    uniq_features = set(feat.values())
    for f in uniq_features:
        os.makedirs(os.path.join(config.TRAIN_DIR, f), exist_ok=True)
        os.makedirs(os.path.join(config.VALIDATION_DIR, f), exist_ok=True)

    identifiers_by_feature = defaultdict(list)
    for f, feature in filtered_feat.items():
        identifiers_by_feature[feature].append(f)

    validation = []
    train = []
    for feature, fs in identifiers_by_feature.items():
        random.shuffle(fs)
        split = int(len(fs) * percent_split)
        validation.extend(fs[:split])
        train.extend(fs[split:])

    random.shuffle(identifiers)
    split = int(len(identifiers) * percent_split)

    for i in train:
        if i in feat:
            copy(files[i], os.path.join(config.TRAIN_DIR, feat[i]))
    for i in validation:
        if i in feat:
            copy(files[i], os.path.join(config.VALIDATION_DIR, feat[i]))


if __name__ == '__main__':
    clear()
    sort()
