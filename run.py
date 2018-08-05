import argparse
import os

from datetime import datetime
from models import models
from db import db, Result
from uuid import uuid4

from keras import backend as K

import numpy as np
import evaluate

from config import config

def test(model, train, validation):
    loss, accuracy = model.evaluate_generator(validation)
    train_loss, train_accuracy = model.evaluate_generator(train)
    results = evaluate.get_results(model, validation)
    probabilities = list(evaluate.transform_binary_probabilities(results))
    labels = list(evaluate.get_labels(validation))

    return {
        "train_accuracy": float(train_accuracy),
        "train_loss": float(train_loss),
        "accuracy": float(accuracy),
        "loss": float(loss),
        "probabilities": probabilities,
        "labels": labels,
    }

def characterize_data(data):
    unique, counts = np.unique(data.classes, return_counts=True)
    index_to_count = dict(zip(unique, counts))
    characterization = { c: index_to_count[data.class_indices[c]] for c in data.class_indices }
    return characterization

def run(model):
    run_id = str(uuid4())

    model.run(run_id)
    K.clear_session()

    model_instance = evaluate.load(os.path.join(
        config.MODEL_DIR,
        "{}-{}.h5".format(run_id, model.MODEL_NAME),
        ))

    train, validation = model.data()
    train_data_stats = characterize_data(train)
    validation_data_stats = characterize_data(validation)
    results = test(model_instance, train, validation)

    result = Result(
        model.MODEL_NAME,
        run_id,
        train_data_stats,
        validation_data_stats,
        **results
        )
    db.session.add(result)
    db.session.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='which model to run (see models.py)')
    parser.add_argument(
        '--trials',
        type=int,
        default=1,
        help='how many times to run')
    FLAGS, unparsed = parser.parse_known_args()
    model = models[FLAGS.model]
    for _ in range(FLAGS.trials):
        run(model)
        K.clear_session()
