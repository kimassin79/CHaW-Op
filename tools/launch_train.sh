set -euo pipefail
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m chaw_op.commands.train "$@"
