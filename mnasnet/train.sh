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

# use ens3 interface for DLAMI Ubuntu and eth0 interface for DLAMI AmazonLinux. If instance type is p3dn.24xlarge, change interface to ens5
INSTANCE_TYPE=`curl http://169.254.169.254/latest/meta-data/instance-type 2>>${CONDA_DEFAULT_ENV}.err`
if [  -n "$(uname -a | grep Ubuntu)" ]; then INTERFACE=ens3; if [ $INSTANCE_TYPE == "p3dn.24xlarge" ]; then INTERFACE=ens5; fi ; else INTERFACE=eth0; fi

~/anaconda3/envs/tensorflow_p36/bin/mpirun -np 8 -hostfile hosts.txt -mca plm_rsh_no_tree_spawn 1 \
        -bind-to socket -map-by slot \
        -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
        -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
        -x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_exclude lo,docker0 \
        -x TF_CPP_MIN_LOG_LEVEL=0 \
        python mnasnet_main_hvd.py --use_tpu=False --data_dir=/home/ubuntu/data --model_dir=./results_hvd --train_batch_size=256 --eval_batch_size=256 --train_steps=218949 --skip_host_call=False --data_format='channels_first' --transpose_input=False --use_horovod=True