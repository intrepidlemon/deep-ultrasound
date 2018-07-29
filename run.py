import argparse

from data import clear, sort
from datetime import datetime
from models import models
from db import db, Result
from uuid import uuid4


def run(model):
    run_id = str(uuid4())

    train_data_stats, validation_data_stats, results = model.run(run_id)

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
    FLAGS, unparsed = parser.parse_known_args()
    model = models[FLAGS.model]
    while True:
        run(model)
