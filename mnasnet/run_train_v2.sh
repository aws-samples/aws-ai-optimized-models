#!/bin/sh

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

export CUDA_VISIBLE_DEVICES="$OMPI_COMM_WORLD_LOCAL_RANK"
echo "using GPU " $OMPI_COMM_WORLD_LOCAL_RANK
python /home/ubuntu/aws-ai-optimized-models/mnasnet/mnasnet_main_hvd_v2.py --use_tpu=False --data_dir=/home/ubuntu/data --model_dir=./results_hvd --train_batch_size=256 --eval_batch_size=256 \
        --train_steps=13684 --steps_per_eval=13684  --iterations_per_loop=13684 --warmup_epochs=60 --base_learning_rate=0.008 --skip_host_call=True --data_format='channels_first' --transpose_input=False --use_horovod=True \
        --eval_on_single_gpu=True --use_larc=False --use_v2=True