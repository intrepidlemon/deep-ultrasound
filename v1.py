import os
from datetime import datetime
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from config import config

MODEL_NAME = "v1"


def data(shuffle_train=True):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary')

    if not shuffle_train:
        train_generator = test_datagen.flow_from_directory(
            config.TRAIN_DIR,
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            batch_size=config.BATCH_SIZE,
            class_mode='binary',
            shuffle=False)

    validation_generator = test_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    return train_generator, validation_generator, test_generator


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
