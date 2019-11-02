# everything with validation and test free images
# pipenv run python data.py --prefix free
pipenv run python run.py --model v4 --trials 100 --description "free-0"

bash notify.sh
