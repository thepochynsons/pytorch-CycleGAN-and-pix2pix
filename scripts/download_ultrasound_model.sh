FILE=$1

echo "available model is aus2rus"

echo "Specified [$FILE]"

URL=https://drive.google.com/file/d/1pEPi0Wmm9iVcE1BsiBpe2IdHGxFICvgR/view?usp=sharing
ZIP_FILE=./checkpoints/$FILE.zip
TARGET_DIR=./checkpoints/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
