# Download dataset
dir=${1:-"data"}
dir=$dir"/VG/"
mkdir -p $dir
echo "Download data to "$dir
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O $dir/images.zip
unzip $dir/images.zip -d $dir/
rm $dir/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O $dir/images2.zip
unzip $dir/images2.zip -d $dir/
rm $dir/images2.zip
