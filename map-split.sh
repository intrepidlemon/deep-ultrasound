# bash map-split.sh ../free ../fixed
# should be run inside the raw directory

mkdir -p $2/train/benign
mkdir -p $2/train/malignant
mkdir -p $2/validation/benign
mkdir -p $2/validation/malignant
mkdir -p $2/test/benign
mkdir -p $2/test/malignant
cp $(ls $1/train/benign | sed 's/free/fixed/g') $2/train/benign
cp $(ls $1/train/malignant | sed 's/free/fixed/g') $2/train/malignant
cp $(ls $1/validation/benign | sed 's/free/fixed/g') $2/validation/benign
cp $(ls $1/validation/malignant | sed 's/free/fixed/g') $2/validation/malignant
cp $(ls $1/test/benign | sed 's/free/fixed/g') $2/test/benign
cp $(ls $1/test/malignant | sed 's/free/fixed/g') $2/test/malignant

