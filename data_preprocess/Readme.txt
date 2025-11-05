###
PETRA: Pretrained Evolutionary TRAnsformer for SARS-CoV-2 Mutation Prediction

Transformer model that directly takes SARS-2 mutation sequences as input and predicts future mutations.

###
1:Data Preparation Phase: Currently, PETRA uses a three-step version for data preprocessing.

1.1: Download SARS-2 phylogenetic tree data. 
Tree providers like UCSC genomics(https://genome.ucsc.edu/) offer downloadable SARS-2 phylogenetic tree data.
Tree based on open database sequences can be found in  https://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/. 
The data preprocessing framework takes taxonium jsonl tree as input. An example can be found in 
https://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/public-latest.all.masked.taxonium.jsonl.gz

For trees including GISAID sequences, according to GISAID rules,
raw GISAID data cannot be directly shared in public at scale. (You can share small amount of data in public, or large scale data with another registered GISAID user.)
Please first apply for an account on GISAID(https://gisaid.org/), 
then ask tree providers for phylogenetic trees including GISAID sequences. Please unzip the data and ensure it is a jsonl.

1.2: Update lineage designation to the most recent ones: 
`` bash update_des.sh ``

1.3: Date preprocessing. Process phylogenetic tree data to model-readable dataset. 
`` bash data_process.sh ``

This processes raw tree data to learnable dataset. It requires some basic python packages that can be easily installed. 

Parameters in data_process.sh that needs to be manually changed:
INPUT_TREE: path of the unzipped full tree json file
DATE: cutoff date of your data
DATA_DIR: path you want to store your processed dataset

Please make sure your device has stable connection with cov-spectrum lapis API(https://lapis.cov-spectrum.org/). 
The program takes in variants from the usher tree designations and verify them according to nextstrain and cov-spectrum. 
It checks all mutations that appear in usher tree designations, and verify if the mutations match the lineage in nextstrain. 
It automatically accepts all mutations that both appears in usher and nextstrain, as well as all deletions on nextstrain because 
usher tree doesn't register deletions.
For mutations that appear only on nextstrain or usher, it verifies them according to cov-spectrum LAPIS. It takes mutations that are
dominant in that variant in cov-spectrum( 10 times more prevalent than its alternative codon). For other mutations, it takes the usher version.

The process will yield you some files.
1: Processed data, stored in DATA_DIR, shuffled and encoded.  
2: processing files. 
Usher_tree+DATE.txt: processed usher tree



