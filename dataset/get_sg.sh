# Download dataset
dir=${1:-"data"}
dir=$dir"/VG/"
mkdir -p $dir
echo "Download data to "$dir
wget https://visualgenome.org/static/data/dataset/scene_graphs.json.zip -O $dir/scene_graphs.zip
unzip $dir/scene_graphs.zip -d $dir/
rm $dir/scene_graphs.zip
