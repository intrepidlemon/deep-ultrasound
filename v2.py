"""
Same as V1, except:
    - class weighting
"""
import os
from datetime import datetime
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np

from config import config

from data import data

MODEL_NAME = "v2"

def model():
    convnet = applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
    )

    # custom Layers
    out = convnet.output
    out = Flatten()(out)
    out = Dense(256, activation="relu")(out)
    out = Dropout(0.5)(out)
    out = Dense(128, activation="relu")(out)
    out = Dropout(0.5)(out)
    out = Dense(64, activation="relu")(out)
    out = Dropout(0.5)(out)
    out = Dense(32, activation="relu")(out)
    out = Dropout(0.5)(out)
    out = Dense(16, activation="relu")(out)
    predictions = Dense(1, activation="sigmoid")(out)

    # creating the final model
    model = Model(inputs=convnet.input, outputs=predictions)

    # compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
        metrics=["accuracy"])

    return model

def class_weight(training):
    unique, counts = np.unique(training.classes, return_counts=True)
    raw_counts = dict(zip(unique, counts))
    return { k: len(training.classes)/v for k, v in raw_counts.items() }

def train(model, training, validation, run_id):
    # save the model according to the conditions
    checkpoint = ModelCheckpoint(
        os.path.join(
            config.MODEL_DIR,
            "{}-{}.h5".format(
                run_id,
                MODEL_NAME,
            ),
        ),
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1,
    )
    early = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=config.PATIENCE,
        verbose=1,
        mode='auto',
    )
    # Train the model
    model.fit_generator(
        training,
        steps_per_epoch=training.n / config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=validation,
        validation_steps=validation.n / config.BATCH_SIZE,
        class_weight=class_weight(training),
        callbacks=[checkpoint, early],
    )

def run(run_id=None):
    if run_id is None:
        run_id = int(datetime.utcnow().timestamp())
    training, validation, _ = data()
    model_instance = model()
    train(model_instance, training, validation, run_id)

if __name__ == '__main__':
    run()
