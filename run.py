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

def test(model, train, validation, test_data):

    loss, accuracy = model.evaluate_generator(validation)
    train_loss, train_accuracy = model.evaluate_generator(train)
    test_loss, test_accuracy = model.evaluate_generator(test_data)

    results = evaluate.get_results(model, validation)
    probabilities = list(evaluate.transform_binary_probabilities(results))
    labels = list(evaluate.get_labels(validation))

    test_results = evaluate.get_results(model, test_data)
    test_probabilities = list(evaluate.transform_binary_probabilities(test_results))
    test_labels = list(evaluate.get_labels(test_data))

    return {
        "train_accuracy": float(train_accuracy),
        "train_loss": float(train_loss),
        "accuracy": float(accuracy),
        "loss": float(loss),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "probabilities": probabilities,
        "labels": labels,
        "test_probabilities": test_probabilities,
        "test_labels": test_labels,
    }

def characterize_data(data):
    unique, counts = np.unique(data.classes, return_counts=True)
    index_to_count = dict(zip(unique, counts))
    characterization = { c: index_to_count[data.class_indices[c]] for c in data.class_indices }
    return characterization

def run(model, description):
    run_id = str(uuid4())

    model.run(run_id)
    K.clear_session()

    model_instance = evaluate.load(os.path.join(
        config.MODEL_DIR,
        "{}-{}.h5".format(run_id, model.MODEL_NAME),
        ))

    train, validation, test_data = model.data()
    train_data_stats = characterize_data(train)
    validation_data_stats = characterize_data(validation)
    test_data_stats = characterize_data(test_data)
    results = test(model_instance, train, validation, test_data)

    result = Result(
        model.MODEL_NAME,
        run_id,
        train_data_stats,
        validation_data_stats,
        test_data_stats,
        description,
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
        '--description',
        type=str,
        help='brief description of the run and its differences')
    parser.add_argument(
        '--trials',
        type=int,
        default=1,
        help='how many times to run')
    FLAGS, unparsed = parser.parse_known_args()
    model = models[FLAGS.model]
    for _ in range(FLAGS.trials):
        run(model, FLAGS.description)
        K.clear_session()
