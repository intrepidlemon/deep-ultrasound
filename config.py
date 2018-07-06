import os
import logging

class Config(object):
    IMAGE_SIZE = 200

    # STEPS_EPOCH * BATCH_SIZE = # training images
    STEPS_EPOCH = 3
    BATCH_SIZE = 16

    EPOCHS = 50
    SAMPLES_VALIDATION = 300

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG

    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")

config = Config()
