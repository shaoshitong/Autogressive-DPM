# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=10043 \
autoregressive/adm_train/extract_codes_c2i_m.py "$@"
