#! /bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
# MASTER_ADDR is the first in SLURM_NODELIST


MASTER_ADDR=localhost
MASTER_PORT=27882


export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mslx5_1:1,mlx5_4:1,mlx5_5:1   
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_DEBUG=INFO


NUM_NODES=1
NODE_RANK=0
echo $NODE_RANK
GPUS_PER_NODE=1



DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)



CHECKPOINT_PATH="" #<Specify path>




GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 1024
    --num-attention-heads 16
    --seq-length 2048
    --max-position-embeddings 2048
    --micro-batch-size 1
    --version 1
)

ROUTE_ARGS=(
    --load $CHECKPOINT_PATH
)



torchrun  ${DISTRIBUTED_ARGS[@]} inference_mutgpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${ROUTE_ARGS[@]} \

