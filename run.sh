# everything with validation and test free images
cp features/all-proof.csv data/features.csv
pipenv run python data.py --prefix free
pipenv run python run.py --model v1 --trials 100 --description "complete-set-free-final-1"
zip -r complete-set-free-$(date +%Y%m%d%H%M%S).zip data

# everything with validation and test fixed images
pipenv run python data.py --prefix fixed
pipenv run python run.py --model v1 --trials 100 --description "complete-set-fixed-final-1"
zip -r complete-set-fixed-$(date +%Y%m%d%H%M%S).zip data

# c3-c4 free images
cp features/c3-c4-proof.csv data/features.csv
pipenv run python data.py --prefix free
pipenv run python run.py --model v2 --trials 100 --description "c3-c4-free-final-1"
zip -r c3-c4-set-free-$(date +%Y%m%d%H%M%S).zip data

# c3-c4 fixed images
pipenv run python data.py --prefix fixed
pipenv run python run.py --model v2 --trials 100 --description "c3-c4-fixed-final-1"
zip -r c3-c4-set-fixed-$(date +%Y%m%d%H%M%S).zip data
