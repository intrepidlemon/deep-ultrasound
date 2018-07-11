import os
import glob
import argparse
import random
import csv
from shutil import copy

from config import config


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


def sort():
    files = all_identifiers(all_files())
    feat = all_features(['benign', 'malignant'])

    identifiers = list(files.keys())
    identifiers = [i for i in identifiers if i in feat]
    random.shuffle(identifiers)
    split = int(len(identifiers) * 0.2)
    validation = identifiers[:split]
    train = identifiers[split:]

    uniq_features = set(feat.values())
    for f in uniq_features:
        os.makedirs(os.path.join(config.TRAIN_DIR, f), exist_ok=True)
        os.makedirs(os.path.join(config.VALIDATION_DIR, f), exist_ok=True)

    for i in train:
        if i in feat:
            copy(files[i], os.path.join(config.TRAIN_DIR, feat[i]))
    for i in validation:
        if i in feat:
            copy(files[i], os.path.join(config.VALIDATION_DIR, feat[i]))


if __name__ == '__main__':
    sort()
