FILE=$1

if [[ $FILE != "aus2rus" ]]; then
    echo "Available dataset is aus2rus"
    exit 1
fi

echo "Specified [$FILE]"
URL='http://download1587.mediafire.com/hnqx5ly535bg/cudar2ah1d2g5t6/aus2rus.zip'
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
