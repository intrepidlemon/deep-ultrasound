set -e

python3.6 setup.py

export FLASK_APP=api.py
flask db init
flask db migrate
flask db upgrade
