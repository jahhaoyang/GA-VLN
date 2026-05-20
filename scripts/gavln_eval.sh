set -x 

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export HYDRA_FULL_ERROR=1

MASTER_IP="YOUR_MASTER_IP"
MASTER_PORT="YOUR_MASTER_PORT"
NODE_NUM="YOUR_NODE_NUM"

OUTPUT="./results/gavln_official/"
CHECKPOINT="./checkpoints/gavln_official/"
VISION_TOWER_PATH="./model/siglip-so400m-patch14-384/"
VGGT_PATH="./model/VGGT-1B/"
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
