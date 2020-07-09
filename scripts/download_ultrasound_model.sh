FILE=$1

echo "available model is aus2rus"

echo "Specified [$FILE]"

URL='set'
ZIP_FILE=./checkpoints/$FILE.zip
TARGET_DIR=./checkpoints/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./checkpoints/
rm $ZIP_FILE
