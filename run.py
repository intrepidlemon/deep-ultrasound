import argparse

from data import clear, sort
from datetime import datetime
from models import models


def run(model):
    run_id = int(datetime.utcnow().timestamp())
    train_loss, train_accuracy, validation_loss, validation_accuracy = model.run(run_id)

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
