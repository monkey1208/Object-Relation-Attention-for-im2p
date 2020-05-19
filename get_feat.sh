wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BiVK-JdxJR-HiH-eoV5663UstSmCVUxQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BiVK-JdxJR-HiH-eoV5663UstSmCVUxQ" -O parabu_fc.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip parabu_fc.zip -d data/bu_data

rm parabu_fc.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uFSLoxfjXs7ltClGxcdfmYzt2RHLRHet' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uFSLoxfjXs7ltClGxcdfmYzt2RHLRHet" -O parabu_att.zip && rm -rf /tmp/cookies.txt
# Unzip the downloaded zip file
unzip parabu_att.zip -d data/bu_data

rm parabu_att.zip
