# bash run.sh <folder name> <description>

rm -rf $DATA_DIR/liver-ultrasound/test
rm -rf $DATA_DIR/liver-ultrasound/validation
rm -rf $DATA_DIR/liver-ultrasound/train
cp -r $DATA_DIR/liver-ultrasound/$1/* $DATA_DIR/liver-ultrasound
pipenv run python run.py --model v2 --trials 100 --description $2

bash notify.sh
