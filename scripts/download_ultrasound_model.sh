FILE=$1

echo "available model is aus2rus"

echo "Specified [$FILE]"

URL='http://download946.mediafire.com/qgregdiu7phg/4n1a535e77pcsdi/kaggle.zip'
ZIP_FILE=./checkpoints/$FILE.zip
TARGET_DIR=./checkpoints/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./checkpoints/
rm $ZIP_FILE
