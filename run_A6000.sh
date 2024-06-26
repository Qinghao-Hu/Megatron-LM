
# A6000 Testing

# torchrun --nproc-per-node 4 examples/run_train.py \
# --batch_size 1 \
# --max_train_steps 30 \
# --seq_length 4_000 \
# --tensor_parallel_size 1 \
# --pipeline_parallel_size 1 \
# --context_parallel_size 4

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=1

CHECKPOINT_PATH=./ #<Specify path>
VOCAB_FILE=./gpt2-vocab.json #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=./gpt2-merges.txt #<Specify path to file>/gpt2-merges.txt
DATA_PATH=./ #<Specify path and file prefix>_text_document


GPUS_PER_NODE=4
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

SEQ_LENGTH_PER_GPU=100
SEQ_LENGTH=$(($WORLD_SIZE*$SEQ_LENGTH_PER_GPU))

TRAINING_DTYPE=bf16
# TRANSFORMER_IMPL=transformer_engine

# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE
#     --nnodes $NUM_NODES
#     --master_addr $MASTER_ADDR
#     --master_port $MASTER_PORT
#     --node_rank $NODE_RANK
# )

# LLaMA-2 7B
GPT_MODEL_ARGS=(
    --num-layers 2
    --hidden-size 4096
    --num-attention-heads 32
    --ffn-hidden-size 11008
    # --activation-func torch.nn.functional.silu
    # --add-bias-linear False
    # --bias_activation_fusion False,
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --apply-query-key-layer-scaling
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $SEQ_LENGTH
    --vocab-size 32000
    --position-embedding-type rope
    --use-flash-attn
    --bf16
    --use-distributed-optimizer
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    # --rampup-batch-size 16 16 5859375
    --train-iters 30
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size $(($GPUS_PER_NODE*$NUM_NODES))
)

DATA_ARGS=(
    --mock-data
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    # --save-interval 10000
    --eval-interval 10000
    --eval-iters 1
    --profile
)


# NSYS Profile
LOG_DIR=.
NSYS_ITER=-1 # -1: off, >0 to enable recommend: 10
NSYS_ITER_RANGE=2

if (( $NSYS_ITER >= 0 )); then
    mkdir -p ${LOG_DIR}/nsys_reports
    NSYS_CMD="/mnt/petrelfs/share/Nsight_Systems/bin/nsys profile --force-overwrite true -o ${LOG_DIR}/nsys_reports/megatron-32GPU-$NODE_RANK --capture-range=cudaProfilerApi"
    NSYS_ARGS="
        --profile --profile-step-start $NSYS_ITER --profile-step-end $(($NSYS_ITER + $NSYS_ITER_RANGE))
    "
else
    NSYS_CMD=""
    NSYS_ARGS=""
fi

$NSYS_CMD torchrun --nproc-per-node 4 pretrain_llama.py \
${GPT_MODEL_ARGS[@]} \
${TRAINING_ARGS[@]} \
${MODEL_PARALLEL_ARGS[@]} \
${DATA_ARGS[@]} \
${EVAL_AND_LOGGING_ARGS[@]} \
${NSYS_ARGS}

# srun -p llm_s --job-name=megatron -n 1 --gres=gpu:8 --ntasks-per-node=1 bash srun.sh
