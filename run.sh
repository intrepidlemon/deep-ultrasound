# everything with validation and test free images
cp features/all-proof.csv data/features.csv
pipenv run python data.py --prefix free
pipenv run python run.py --model v1 --trials 100 --description "complete-set-free-final"
zip -r complete-set-free.zip data

# everything with validation and test fixed images
pipenv run python data.py --prefix fixed
pipenv run python run.py --model v1 --trials 100 --description "complete-set-fixed-final"
zip -r complete-set-fixed.zip data
# c3-c4 free images
rm data/test.csv
touch data/test.csv
cp features/c3-c4-proof.csv data/features.csv
pipenv run python data.py --prefix free
pipenv run python run.py --model v2 --trials 100 --description "c3-c4-free-final"
zip -r c3-c4-free.zip data
# c3-c4 fixed images
pipenv run python data.py --prefix fixed
pipenv run python run.py --model v2 --trials 100 --description "c3-c4-fixed-final"
zip -r c3-c4-fixed.zip data

curl --request POST \
  --url https://api.sendgrid.com/v3/mail/send \
  --header "authorization: Bearer $SENDGRID_API_KEY" \
  --header 'content-type: application/json' \
  --data '{"personalizations":[{"to":[{"email":"iantolinxi@gmail.com","name":"Ianto Xi"}],"subject":"process complete"}],"from":{"email":"1080@harrisonbai.com","name":"1080 Harrison Bai"},"reply_to":{"email":"no-repla
y@harrisonbai.com","name":"no reply"}, "content": [{"type": "text/plain", "value": "process done"}]}'


