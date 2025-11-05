#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

source .bashrc
module load cuda/11.8

MASTER_PORT=16000
export WORLD_SIZE=32
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR




CHECKPOINT_PATH="/share/home/zouxu/ckpts_sample" #<Specify path>
TENSORBOARD_LOGS_PATH="/share/home/zouxu/logs" #<Specify path>
VOCAB_FILE="/share/home/zouxu/gpt2-vocab.json" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="/share/home/zouxu/gpt2-merges.txt" #<Specify path to file>/gpt2-merges.txt
DATA_PATH="/share/home/zouxu/data_sample_new_text_document" #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 512
    --num-attention-heads 8
    --seq-length 1024
    --max-position-embeddings 1024
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 64
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 2000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

srun /share/home/zouxu/Megatron-LM/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
