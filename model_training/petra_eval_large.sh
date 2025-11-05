#! /bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
# MASTER_ADDR is the first in SLURM_NODELIST

MASTER_ADDR="127.0.0.1"
MASTER_PORT=27888
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=27888

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mslx5_1:1,mlx5_4:1,mlx5_5:1   
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_DEBUG=INFO


NUM_NODES=1
NODE_RANK=0
echo $NODE_RANK
GPUS_PER_NODE=8



DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)
CHECKPOINT_PATH="" #<Specify path>
TENSORBOARD_LOGS_PATH="" #<Specify path>
OUTPUT_FILE=""


echo $MASTER_ADDR
echo $MASTER_PORT


GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 1024
    --num-attention-heads 16
    --seq-length 2048
    --max-position-embeddings 2048
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --train-iters 60000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 1.0e-4
    --lr-decay-style cosine 
    --min-lr 1.0e-5
    --lr-warmup-fraction .001 
    --lr-decay-iters 50000 
    --valid-dataset 2025-07-16
    --valid-start-date 2025-02-13
    --version 1
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)


EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 3000
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --ckpt-step 60000
    --eval-iters 100
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --output-file $OUTPUT_FILE
)

torchrun  ${DISTRIBUTED_ARGS[@]} eval_mutgpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
