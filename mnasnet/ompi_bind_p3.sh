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

#!/bin/bash

case "${OMPI_COMM_WORLD_LOCAL_RANK}" in
0)
    export OMPI_MCA_btl_openib_if_include=mlx5_0
    exec numactl --physcpubind=0-3,32-35 --membind=0 "${@}"
    ;;
1)
    export OMPI_MCA_btl_openib_if_include=mlx5_0
    exec numactl --physcpubind=4-7,36-39 --membind=0 "${@}"
    ;;
2)
    export OMPI_MCA_btl_openib_if_include=mlx5_1
    exec numactl --physcpubind=8-11,40-43 --membind=0 "${@}"
    ;;
3)
    export OMPI_MCA_btl_openib_if_include=mlx5_1
    exec numactl --physcpubind=12-15,44-47 --membind=0 "${@}"
    ;;
4)
    export OMPI_MCA_btl_openib_if_include=mlx5_2
    exec numactl --physcpubind=16-19,48-51 --membind=1 "${@}"
    ;;
5)
    export OMPI_MCA_btl_openib_if_include=mlx5_2
    exec numactl --physcpubind=20-23,52-55 --membind=1 "${@}"
    ;;
6)
    export OMPI_MCA_btl_openib_if_include=mlx5_3
    exec numactl --physcpubind=24-27,56-59 --membind=1 "${@}"
    ;;
7)
    export OMPI_MCA_btl_openib_if_include=mlx5_3
    exec numactl --physcpubind=28-31,60-63 --membind=1 "${@}"
    ;;
*)
    echo ==============================================================
    echo "ERROR: Unknown local rank ${OMPI_COMM_WORLD_LOCAL_RANK}"
    echo ==============================================================
    exit 1
    ;;
esac