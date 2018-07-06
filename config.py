import os
import logging

class Config(object):
    IMAGE_SIZE = 200

    BATCH_SIZE = 1

    EPOCHS = 50
    SAMPLES_VALIDATION = 300

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG

    OUTPUT = "output"
    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")

config = Config()
