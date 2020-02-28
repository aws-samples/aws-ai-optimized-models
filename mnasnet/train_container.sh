# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env bash
NUM_GPU=${1:-1}
TRAIN_STEPS=${2:-218949}
STEPS_PER_EVAL=${3:-218949}
TRAIN_BATCH_SIZE=${4:-256}
EVAL_BATCH_SIZE=${5:-256}

mpirun -np ${NUM_GPU}  -mca plm_rsh_no_tree_spawn 1 \
        -bind-to none -map-by slot \
	-x TF_ENABLE_AUTO_MIXED_PRECISION=1 \
        -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
        -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
        -x NCCL_SOCKET_IFNAME=^docker0,lo -mca btl_tcp_if_exclude lo,docker0 \
        -x TF_CPP_MIN_LOG_LEVEL=0 --allow-run-as-root \
        -x TF_ENABLE_NHWC=1 \
        -x TF_CUDNN_USE_AUTOTUNE=1 \
        -x TF_ENABLE_XLA=0 \
        python3 /aws-ai-optimized-models/mnasnet/mnasnet_main_hvd.py --use_tpu=False \
        --data_dir=/fsx/mnasnet --model_dir=./results_hvd \
        --train_batch_size=${TRAIN_BATCH_SIZE} --eval_batch_size=${EVAL_BATCH_SIZE} \
        --train_steps=${TRAIN_STEPS} --steps_per_eval=${STEPS_PER_EVAL} --skip_host_call=True --data_format='channels_first' \
        --transpose_input=False --use_horovod=True --num_parallel_calls=64 --eval_on_single_gpu=True \
        --use_larc=False
