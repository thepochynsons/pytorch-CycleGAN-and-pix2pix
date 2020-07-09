FILE=$1

if [[ $FILE != "aus2rus" ]]; then
    echo "Available dataset is aus2rus"
    exit 1
fi

echo "Specified [$FILE]"
URL='set'
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
