config_file=$1
num_gpus=$2
accelerate_config_file=$3

# train.py resolves configs/, ckpts/ and neighborhoods/ relative to src/, while
# models/qwen2_5_omni.py imports `src.models.*`, so run from src/ with the repo
# root on PYTHONPATH. Extra args (e.g. --debug) are forwarded to train.py.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT/src" || exit 1

# The sheet layout is read from $TOPO_SHEET_LAYOUT at IMPORT time in spatial_utils -- it
# also fixes the adaptor width and whether the encoders build a front-end adaptor at all --
# so a key inside the config would be read far too late. Lift it out of the config here.
# Configs without a `sheet-layout:` key leave the variable exactly as the caller set it,
# so every pre-existing launch behaves as before.
_sheet_layout=$(sed -n 's/^sheet-layout:[[:space:]]*\([^[:space:]#]*\).*/\1/p' \
                "configs/${config_file}" 2>/dev/null | head -1)
if [ -n "$_sheet_layout" ]; then
    export TOPO_SHEET_LAYOUT="$_sheet_layout"
    echo "> TOPO_SHEET_LAYOUT=$TOPO_SHEET_LAYOUT (from configs/${config_file})"
fi

# the spatial loss allocates many differently-sized (K, K) blocks per step;
# expandable segments keeps that from fragmenting the large logits allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --num_processes $num_gpus \
    --config_file configs/${accelerate_config_file}  \
    training/train.py -c $config_file \
    --wandb "${@:4}"
