import os
import logging

DATA_DIR = os.environ["DATA_DIR"]

class Config(object):
    IMAGE_SIZE = 200

    BATCH_SIZE = 16

    PATIENCE = 50
    EPOCHS = 500

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG

    DATA = DATA_DIR
    RAW_DIR = os.path.join(DATA, "segmented")
    TRAIN_DIR = os.path.join(DATA, "train")
    TEST_DIR = os.path.join(DATA, "test")
    VALIDATION_DIR = os.path.join(DATA, "validation")
    FEATURES = os.path.join(DATA, "features.csv")
    UNCERTAIN_FEATURES = os.path.join(DATA, "c3-c4-features.csv")
    EXPERTS = os.path.join(DATA, "experts.csv")
    UNCERTAIN_EXPERTS = os.path.join(DATA, "c3-c4-experts.csv")
    TEST_SET = os.path.join(DATA, "test.csv")
    UNCERTAIN_TEST_SET = os.path.join(DATA, "c3-c4-test.csv")

    OUTPUT = DATA_DIR
    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")

config = Config()
