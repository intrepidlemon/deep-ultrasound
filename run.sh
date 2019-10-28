# everything with validation and test free images
pipenv run python data.py --prefix free
pipenv run python run.py --model v2 --trials 100 --description "free-0"

# everything with validation and test fixed images
pipenv run python data.py --prefix fixed
pipenv run python run.py --model v2 --trials 100 --description "fixed-0"

bash notify.sh
