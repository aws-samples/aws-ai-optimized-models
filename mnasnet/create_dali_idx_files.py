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

from subprocess import call
import os.path
import glob

tfrecord_files = glob.glob('/data/*')
if not os.path.exists("index_files"):
    os.mkdir("index_files")
tfrecord2idx = "tfrecord2idx"
for tfrecord in tfrecord_files:
    tfrecord_idx = 'index_files' + '/' + os.path.basename(tfrecord) + '.idx'
    if not os.path.isfile(tfrecord_idx):
        call([tfrecord2idx, tfrecord, tfrecord_idx])


#tfrecord = "/data/train-00001-of-01024"
#tfrecord_idx = "/data/index_files/train-00001-of-01024.idx"
