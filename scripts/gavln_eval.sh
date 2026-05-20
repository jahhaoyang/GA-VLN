set -x 

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_PATH=$CUDA_HOME/bin:$CUDA_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export HYDRA_FULL_ERROR=1

MASTER_IP=localhost
MASTER_PORT=23360
NODE_NUM=4

OUTPUT="/ssd1/jhy_/results/gavln_official/"
CHECKPOINT="/ssd1/jhy_/checkpoints/gavln_official/"
VISION_TOWER_PATH="/ssd1/jhy/model/siglip-so400m-patch14-384/"
VGGT_PATH="/ssd1/jhy/model/VGGT-1B/"
echo "CHECKPOINT: ${CHECKPOINT}"

torchrun --nnodes=1 \
    --nproc_per_node=$NODE_NUM \
    --node_rank=0 \
    --master_port=$MASTER_PORT \
    --master_addr $MASTER_IP \
    gavln/gavln_eval.py \
    --model_path $CHECKPOINT \
    --vision_tower_path $VISION_TOWER_PATH \
    --vggt_path $VGGT_PATH \
    --output_path $OUTPUT
