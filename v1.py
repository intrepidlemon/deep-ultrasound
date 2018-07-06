from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from config import config

def data(directory="data"):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            '{}/train'.format(directory),
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            batch_size=config.BATCH_SIZE,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            '{}/validation'.format(directory),
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            batch_size=config.BATCH_SIZE,
            class_mode='binary')

    return train_generator, validation_generator

def model():
    convnet = applications.ResNet50(
            weights = "imagenet",
            include_top=False,
            input_shape = (config.IMAGE_SIZE, config.IMAPGE_SIZE, 3),
            )
    # fix convnet weights to imagenet
    for layer in convnet.layers:
        layer.trainable = False

    # custom Layers
    out = convnet.output
    out = Flatten()(out)
    out = Dense(1024, activation="relu")(out)
    out = Dropout(0.5)(out)
    out = Dense(1024, activation="relu")(out)
    predictions = Dense(16, activation="softmax")(out)

    # creating the final model
    model = Model(input = convnet.input, output = predictions)

    # compile the model
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = optimizers.SGD(lr=0.0001, momentum=0.9),
        metrics=["accuracy"]
        )

    return model

def train(model, training, validation):
    # save the model according to the conditions
    checkpoint = ModelCheckpoint(
            "vgg16_1.h5",
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
            patience=10,
            verbose=1,
            mode='auto',
            )
    # Train the model
    model.fit_generator(
        training,
        samples_per_epoch = training.nb_samples,
        epochs = config.EPOCHS,
        validation_data = validation,
        nb_val_samples = validation.nb_samples,
        callbacks = [checkpoint, early],
    )

def run():
    training, validation = data()
    model = mode()
    train(model, training, validation)



