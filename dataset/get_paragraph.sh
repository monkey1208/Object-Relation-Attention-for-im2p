# Download dataset
dir=${1:-"data"}
echo "Download data to "$dir
wget http://visualgenome.org/static/data/dataset/paragraphs_v1.json.zip -O $dir/paragraphs_v1.json.zip
# Unzip the downloaded zip file
unzip $dir/paragraphs_v1.json.zip -d $dir/

# Remove the downloaded zip file
rm $dir/paragraphs_v1.json.zip
