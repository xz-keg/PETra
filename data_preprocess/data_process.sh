INPUT_TREE=".."  # requires GISAID account and get from usher
DATA_DIR="../data2025-07-16"
DATE="2025-07-16"

python process_usher.py --date $DATE --input-tree $INPUT_TREE
python process_dataset.py --date $DATE
python lr_gather.py --date $DATE
python process_diff.py --date $DATE
python query_difference.py --date $DATE
python process_encode.py --date $DATE --data-dir $DATA_DIR
python shuffle_data.py --date $DATE --data-dir $DATA_DIR

