###
Model training (Code based on nvidia Megatron-LM)

Once you processed the data, you can start to train the model. Ensure you have a GPU server for training and evaluation. 
You can adjust batch size or related parameters to suit your own GPU device. 


``bash petra.sh``

Please change the "default_path" in ``pretrain_mutgpt_batched.py`` to the directory you put your data.
Please change the "CHECKPOINT_PATH" in ``petra.sh`` to the place you want to put your checkpoint. 

Also please change the batch-size related configurations depending on your GPU device.

This allows you to train the model given the usher tree updated on 2025-2-12. 
It uses the usher tree on 2025-7-16 as default evaluation data. 
It evaluates only on sequences collected after 2025-2-13 so as to maintain the predictive nature. 

For evaluation, use

``bash eval_petra.sh``

This evaluates the performance and outputs the result in "OUTPUT_FILE"

Also, please change the "default_path" in ``eval_mutgpt.py`` and "CHECKPOINT_PATH" in ``eval_petra.sh``.

For other experiments, we also offer code to train and evaluate petra-large and also evaluate by month.

Also, some supplementary codes are also included in this repo. Please adjust the "data_dir_prefix" to the parent directory of your processed usher tree data before using them.

``python country_distribute.py``  for generate a sequence statistics for developed, developing and least developed countries.
``python evaluate_bloom.py`` for evaluation of Bloom estimator. Bloom fitness scores for each mutation can be downloaded directly from (https://github.com/jbloomlab/SARS2-mut-fitness)
``python data_stat.py`` for dataset-level statistics. 

