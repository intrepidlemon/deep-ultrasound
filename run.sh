# everything with validation and test free images
rm -rf $DATA_DIR/liver-ultrasound/test
rm -rf $DATA_DIR/liver-ultrasound/validation
rm -rf $DATA_DIR/liver-ultrasound/train
cp -r $DATA_DIR/liver-ultrasound/free/* $DATA_DIR/liver-ultrasound
pipenv run python run.py --model v2 --trials 100 --description "free-0"

rm -rf $DATA_DIR/liver-ultrasound/test
rm -rf $DATA_DIR/liver-ultrasound/validation
rm -rf $DATA_DIR/liver-ultrasound/train
cp -r $DATA_DIR/liver-ultrasound/fixed/* $DATA_DIR/liver-ultrasound
pipenv run python run.py --model v2 --trials 100 --description "fixed-0"

bash notify.sh
