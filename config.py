import os
import logging


class Config(object):
    IMAGE_SIZE = 200

    BATCH_SIZE = 128

    EPOCHS = 50

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG

    DATA = "data"
    RAW_DIR = os.path.join(DATA, "raw")
    TRAIN_DIR = os.path.join(DATA, "train")
    TEST_DIR = os.path.join(DATA, "test")
    VALIDATION_DIR = os.path.join(DATA, "validation")
    FEATURES = os.path.join(DATA, "features.csv")

    OUTPUT = "output"
    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")


config = Config()
