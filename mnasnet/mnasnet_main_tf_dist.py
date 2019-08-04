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

# Original license
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a MnasNet on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

import imagenet_input
import mnasnet_models
import mnasnet_utils
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator
from tensorflow.python.keras import backend as K
import json

FLAGS = flags.FLAGS

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

flags.DEFINE_bool(
    'use_tpu',
    default=True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific flags
flags.DEFINE_string(
    'data_dir',
    default=FAKE_DATA_DIR,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'model_dir',
    default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_string(
    'model_name',
    default='mnasnet-a1',
    help=(
        'The model name to select models among existing MnasNet configurations.'
    ))

flags.DEFINE_string(
    'mode',
    default='train_and_eval',
    help='One of {"train_and_eval", "train", "eval"}.')

flags.DEFINE_integer(
    'train_steps',
    default=437898, #for global bs 1024
    help=('The number of steps to use for training. Default is 437898 steps'
          ' which is approximately 350 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer('input_image_size', default=224, help='Input image size.')

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'steps_per_eval',
    default=6255, #780
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_bool(
    'skip_host_call',
    default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop',
    default=2560, #6255, # mnasnet opt - modified this since eval is triggered in tf dist based on this
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_parallel_calls',
    default=64,
    help=('Number of parallel threads in CPU for the input pipeline'))

flags.DEFINE_string(
    'bigtable_project', None,
    'The Cloud Bigtable project.  If None, --gcp_project will be used.')
flags.DEFINE_string('bigtable_instance', None,
                    'The Cloud Bigtable instance to load data from.')
flags.DEFINE_string('bigtable_table', 'imagenet',
                    'The Cloud Bigtable table to load data from.')
flags.DEFINE_string('bigtable_train_prefix', 'train_',
                    'The prefix identifying training rows.')
flags.DEFINE_string('bigtable_eval_prefix', 'validation_',
                    'The prefix identifying evaluation rows.')
flags.DEFINE_string('bigtable_column_family', 'tfexample',
                    'The column family storing TFExamples.')
flags.DEFINE_string('bigtable_column_qualifier', 'example',
                    'The column name storing TFExamples.')

flags.DEFINE_string(
    'data_format',
    default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))
flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')
flags.DEFINE_float(
    'batch_norm_momentum',
    default=None,
    help=('Batch normalization layer momentum of moving average to override.'))
flags.DEFINE_float(
    'batch_norm_epsilon',
    default=None,
    help=('Batch normalization layer epsilon to override..'))

flags.DEFINE_bool(
    'transpose_input',
    default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))
flags.DEFINE_bool(
    'export_to_tpu',
    default=False,
    help=('Whether to export additional metagraph with "serve, tpu" tags'
          ' in addition to "serve" only metagraph.'))
flags.DEFINE_bool(
    'post_quantize', default=True, help=('Enable post quantization.'))

flags.DEFINE_bool(
    'quantized_training',
    default=False,
    help=('Enable quantized training as it is required for Edge TPU.'
          'This should be used for fine-tuning rather than pre-training.'))

flags.DEFINE_integer(
    'quantization_delay_epochs',
    default=0,
    help=('The number of epochs after which weights and activations are'
          ' quantized during training.'))

flags.DEFINE_bool(
    'export_moving_average',
    default=False,
    help=('Replace variables with corresponding moving average variables in '
          'saved model export.'))

flags.DEFINE_string(
    'init_checkpoint',
    default=None,
    help=('Initial checkpoint from a pre-trained MnasNet model.'))

flags.DEFINE_float(
    'base_learning_rate',
    default=0.016,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum',
    default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'moving_average_decay', default=0.9999, help=('Moving average decay rate.'))

flags.DEFINE_float(
    'weight_decay',
    default=1e-5,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing',
    default=0.1,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_float(
    'dropout_rate',
    default=0.2,
    help=('Dropout rate for the final output layer.'))

flags.DEFINE_integer(
    'log_step_count_steps', 64, 'The number of steps at '
    'which the global step information is logged.')

flags.DEFINE_bool(
    'use_cache', default=True, help=('Enable cache for training input.'))

flags.DEFINE_float(
    'depth_multiplier', default=None, help=('Depth multiplier per layer.'))

flags.DEFINE_float(
    'depth_divisor', default=None, help=('Depth divisor (default to 8).'))

flags.DEFINE_float(
    'min_depth', default=None, help=('Minimal depth (default to None).'))

flags.DEFINE_bool(
    'use_async_checkpointing', default=False, help=('Enable async checkpoint'))

flags.DEFINE_bool(
    'use_bfloat16',
    default=False,
    help=('Whether to use bfloat16 as activation for training.'))

flags.DEFINE_bool(
    'use_keras',
    default=True,
    help=('Whether to use tf.keras.layers to construct networks.'))

flags.DEFINE_bool(
    'use_xla',
    default=False,
    help=('Whether to use xla.'))

flags.DEFINE_bool(
    'eval_on_single_gpu',
    default=False,
    help=('Whether to eval on single gpu. If false, evaluation is performed on all available gpus'))

flags.DEFINE_integer(
    'warmup_epochs', 5, 'The number of warmup epochs to ramp up lr')

flags.DEFINE_integer(
    'total_nodes', 1, 'The total number of nodes for TF dist using mirrored strategy')

flags.DEFINE_integer(
    'node_num', 0, 'The node used for TF dist using mirrored strategy')

flags.DEFINE_bool(
    'is_evaluator',
    default=False,
    help=('Use this node as evaluator'))


# Learning rate schedule
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def get_pretrained_variables_to_restore(checkpoint_path,
                                        load_moving_average=False):
  """Gets veriables_to_restore mapping from pretrained checkpoint.

  Args:
    checkpoint_path: String. Path of checkpoint.
    load_moving_average: Boolean, whether load moving average variables to
      replace variables.

  Returns:
    Mapping of variables to restore.
  """
  checkpoint_reader = tf.train.load_checkpoint(checkpoint_path)
  variable_shape_map = checkpoint_reader.get_variable_to_shape_map()

  variables_to_restore = {}
  for v in tf.global_variables():
    # Skip variables if they are in excluded scopes.
    is_excluded = False
    for scope in ['global_step', 'ExponentialMovingAverage']:
      if scope in v.op.name:
        is_excluded = True
        break
    if is_excluded:
      tf.logging.info('Exclude [%s] from loading from checkpoint.', v.op.name)
      continue
    variable_name_ckpt = v.op.name
    if load_moving_average:
      # To load moving average variables into non-moving version for
      # fine-tuning, maps variables here manually.
      variable_name_ckpt = v.op.name + '/ExponentialMovingAverage'
      if variable_name_ckpt not in variable_shape_map:
        tf.logging.info(
            'Skip init [%s] from [%s] as it is not in the checkpoint',
            v.op.name, variable_name_ckpt)
        continue

    variables_to_restore[variable_name_ckpt] = v
    tf.logging.info('Init variable [%s] from [%s] in ckpt', v.op.name,
                    variable_name_ckpt)
  return variables_to_restore

#gstep = tf.train.get_global_step()
ema = tf.train.ExponentialMovingAverage(decay=0.999) #, num_updates=gstep)

def mnasnet_model_fn(features, labels, mode, params):
  """The model_fn for MnasNet to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
      `params['batch_size']` is always provided and should be used as the
      effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  # This is essential, if using a keras-derived model.
  K.set_learning_phase(is_training)

  if isinstance(features, dict):
    features = features['feature']

  # In most cases, the default data format NCHW instead of NHWC should be
  # used for a significant performance boost on GPU/TPU. NHWC should be used
  # only if the network needs to be run on CPU since the pooling operations
  # are only supported on NHWC.
  if FLAGS.data_format == 'channels_first':
    assert not FLAGS.transpose_input    # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])

  if FLAGS.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  # Normalize the image to zero mean and unit variance.
  if FLAGS.data_format == 'channels_last':
    features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
    features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
  else:
    features -= tf.constant(MEAN_RGB, shape=[3, 1, 1], dtype=features.dtype)
    features /= tf.constant(STDDEV_RGB, shape=[3, 1, 1], dtype=features.dtype)
  
  has_moving_average_decay = (FLAGS.moving_average_decay > 0)

  tf.logging.info('Using open-source implementation for MnasNet definition.')
  override_params = {}
  if FLAGS.batch_norm_momentum:
    override_params['batch_norm_momentum'] = FLAGS.batch_norm_momentum
  if FLAGS.batch_norm_epsilon:
    override_params['batch_norm_epsilon'] = FLAGS.batch_norm_epsilon
  if FLAGS.dropout_rate:
    override_params['dropout_rate'] = FLAGS.dropout_rate
  if FLAGS.data_format:
    override_params['data_format'] = FLAGS.data_format
  if FLAGS.num_label_classes:
    override_params['num_classes'] = FLAGS.num_label_classes
  if FLAGS.depth_multiplier:
    override_params['depth_multiplier'] = FLAGS.depth_multiplier
  if FLAGS.depth_divisor:
    override_params['depth_divisor'] = FLAGS.depth_divisor
  if FLAGS.min_depth:
    override_params['min_depth'] = FLAGS.min_depth
  override_params['use_keras'] = FLAGS.use_keras

  if params['use_bfloat16']:
    with tf.contrib.tpu.bfloat16_scope():
      logits, _ = mnasnet_models.build_mnasnet_model(
          features,
          model_name=FLAGS.model_name,
          training=is_training,
          override_params=override_params)
    logits = tf.cast(logits, tf.float32)
  else:
    logits, _ = mnasnet_models.build_mnasnet_model(
        features,
        model_name=FLAGS.model_name,
        training=is_training,
        override_params=override_params)

  if params['quantized_training']:
    if is_training:
      tf.logging.info('Adding fake quantization ops for training.')
      tf.contrib.quantize.create_training_graph(
          quant_delay=int(params['steps_per_epoch'] *
                          FLAGS.quantization_delay_epochs))
    else:
      tf.logging.info('Adding fake quantization ops for evaluation.')
      tf.contrib.quantize.create_eval_graph()

  if mode == tf.estimator.ModeKeys.PREDICT:
    scaffold_fn = None
    if FLAGS.export_moving_average:
      restore_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
      variables_to_restore = get_pretrained_variables_to_restore(
          restore_checkpoint, load_moving_average=True)
      tf.logging.info('Restoring from the latest checkpoint: %s',
                      restore_checkpoint)
      tf.logging.info(str(variables_to_restore))

      def restore_scaffold():
        saver = tf.train.Saver(variables_to_restore)
        return tf.train.Scaffold(saver=saver)

      scaffold_fn = restore_scaffold

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        },
        scaffold_fn=scaffold_fn)

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  params['batch_size'] = FLAGS.train_batch_size
  batch_size = params['batch_size']   # pylint: disable=unused-variable

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, FLAGS.num_label_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=FLAGS.label_smoothing)

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = cross_entropy + FLAGS.weight_decay * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.trainable_variables()
      if 'batch_normalization' not in v.name
  ])

  global_step = tf.train.get_global_step()
  if has_moving_average_decay:
    # ema = tf.train.ExponentialMovingAverage(
    #     decay=FLAGS.moving_average_decay, num_updates=global_step)
    ema._num_updates = global_step
    ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
    for v in tf.global_variables():
      # We maintain mva for batch norm moving mean and variance as well.
      if 'moving_mean' in v.name or 'moving_variance' in v.name:
        ema_vars.append(v)
    ema_vars = list(set(ema_vars))

  host_call = None
  if is_training:
    # Compute the current epoch and associated learning rate from global_step.
    current_epoch = (
        tf.cast(global_step, tf.float32) / params['steps_per_epoch'])

    scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)
    learning_rate = mnasnet_utils.build_learning_rate(scaled_lr, global_step,
                                                      params['steps_per_epoch'], warmup_epochs=FLAGS.warmup_epochs)
   
    optimizer = mnasnet_utils.build_optimizer(learning_rate)
    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #   train_op = optimizer.minimize(loss, global_step)
    minimize_op = optimizer.minimize(loss=loss, global_step=global_step)
    train_op = tf.group(minimize_op, update_ops)

    # if has_moving_average_decay:
    #   with tf.control_dependencies([train_op]):
    #     train_op = ema.apply(ema_vars)
    if has_moving_average_decay:
        ema_op = ema.apply(ema_vars)
        train_op = tf.group(ema_op, [train_op])

    if not FLAGS.skip_host_call:

      def host_call_fn(gs, loss, lr, ce):
        """Training host call.

        Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with tf.contrib.summary.create_file_writer(
            FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
          with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', loss[0], step=gs)
            tf.contrib.summary.scalar('learning_rate', lr[0], step=gs)
            tf.contrib.summary.scalar('current_epoch', ce[0], step=gs)

            return tf.contrib.summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(loss, [1])
      lr_t = tf.reshape(learning_rate, [1])
      ce_t = tf.reshape(current_epoch, [1])

      host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

  else:
    train_op = None

  eval_metrics = {}
  if mode == tf.estimator.ModeKeys.EVAL:
    predictions = tf.argmax(logits, axis=1)
    top_1_accuracy = tf.metrics.accuracy(labels, predictions)
    in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
    top_5_accuracy = tf.metrics.mean(in_top_5)
    eval_metrics = {'top_1_accuracy': top_1_accuracy,'top_5_accuracy': top_5_accuracy}

  num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
  tf.logging.info('number of trainable parameters: {}'.format(num_params))

  # Prepares scaffold_fn if needed.
  scaffold_fn = None
  if is_training and FLAGS.init_checkpoint:
    variables_to_restore = get_pretrained_variables_to_restore(
        FLAGS.init_checkpoint, has_moving_average_decay)
    tf.logging.info('Initializing from pretrained checkpoint: %s',
                    FLAGS.init_checkpoint)
    if FLAGS.use_tpu:

      def init_scaffold():
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint,
                                      variables_to_restore)
        return tf.train.Scaffold()

      scaffold_fn = init_scaffold
    else:
      tf.train.init_from_checkpoint(FLAGS.init_checkpoint, variables_to_restore)

  restore_vars_dict = None
  if not is_training and has_moving_average_decay:
    # Load moving average variables for eval.
    restore_vars_dict = ema.variables_to_restore(ema_vars)

    #def eval_scaffold():
    saver = tf.train.Saver(restore_vars_dict)
    scaffold_fn = tf.train.Scaffold(saver=saver)

    #scaffold_fn = eval_scaffold

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics,
      scaffold=scaffold_fn)


def _verify_non_empty_string(value, field_name):
  """Ensures that a given proposed field value is a non-empty string.

  Args:
    value:  proposed value for the field.
    field_name:  string name of the field, e.g. `project`.

  Returns:
    The given value, provided that it passed the checks.

  Raises:
    ValueError:  the value is not a string, or is a blank string.
  """
  if not isinstance(value, str):
    raise ValueError('Bigtable parameter "%s" must be a string.' % field_name)
  if not value:
    raise ValueError('Bigtable parameter "%s" must be non-empty.' % field_name)
  return value


def _select_tables_from_flags():
  """Construct training and evaluation Bigtable selections from flags.

  Returns:
    [training_selection, evaluation_selection]
  """
  project = _verify_non_empty_string(
      FLAGS.bigtable_project or FLAGS.gcp_project, 'project')
  instance = _verify_non_empty_string(FLAGS.bigtable_instance, 'instance')
  table = _verify_non_empty_string(FLAGS.bigtable_table, 'table')
  train_prefix = _verify_non_empty_string(FLAGS.bigtable_train_prefix,
                                          'train_prefix')
  eval_prefix = _verify_non_empty_string(FLAGS.bigtable_eval_prefix,
                                         'eval_prefix')
  column_family = _verify_non_empty_string(FLAGS.bigtable_column_family,
                                           'column_family')
  column_qualifier = _verify_non_empty_string(FLAGS.bigtable_column_qualifier,
                                              'column_qualifier')
  return [
      imagenet_input.BigtableSelection(
          project=project,
          instance=instance,
          table=table,
          prefix=p,
          column_family=column_family,
          column_qualifier=column_qualifier)
      for p in (train_prefix, eval_prefix)
  ]


def export(est, export_dir, post_quantize=True):
  """Export graph to SavedModel and TensorFlow Lite.

  Args:
    est: estimator instance.
    export_dir: string, exporting directory.
    post_quantize: boolean, whether to quantize model checkpoint after training.

  Raises:
    ValueError: the export directory path is not specified.
  """
  if not export_dir:
    raise ValueError('The export directory path is not specified.')
  # The guide to serve a exported TensorFlow model is at:
  #    https://www.tensorflow.org/serving/serving_basic
  image_serving_input_fn = imagenet_input.build_image_serving_input_fn(
      FLAGS.input_image_size)

  tf.logging.info('Starting to export model.')
  subfolder = est.export_saved_model(
      export_dir_base=export_dir,
      serving_input_receiver_fn=image_serving_input_fn)

  tf.logging.info('Starting to export TFLite.')
  converter = tf.lite.TFLiteConverter.from_saved_model(
      subfolder, input_arrays=['truediv'], output_arrays=['logits'])
  tflite_model = converter.convert()
  tflite_file = os.path.join(export_dir, FLAGS.model_name + '.tflite')
  tf.gfile.GFile(tflite_file, 'wb').write(tflite_model)

  if post_quantize:
    tf.logging.info('Starting to export quantized TFLite.')
    converter = tf.lite.TFLiteConverter.from_saved_model(
        subfolder, input_arrays=['truediv'], output_arrays=['logits'])
    converter.post_training_quantize = True
    quant_tflite_model = converter.convert()
    quant_tflite_file = os.path.join(export_dir,
                                     FLAGS.model_name + '_postquant.tflite')
    tf.gfile.GFile(quant_tflite_file, 'wb').write(quant_tflite_model)


def main(unused_argv):
  # Mnas optimize - set the proper image data format
  tf.keras.backend.set_image_data_format(FLAGS.data_format)
  # Mnas optimize - optimization flags
  # gpu_thread_count = 2
  # os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
  # os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
  # os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  # os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  # enable mixed precision? -> Not much benefits seen yet
  # os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  
  node0 = "172.31.11.9:6060"
  node1 = "172.31.1.33:6060"

  strategy = tf.distribute.MirroredStrategy() 
  if FLAGS.total_nodes > 1:
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
    if not FLAGS.is_evaluator:
      if FLAGS.node_num == 0:
        os.environ['TF_CONFIG'] = json.dumps({
          'cluster': {
              'worker': [node0, node1]
          },
          'task': {'type': 'worker', 'index': 0}
        })
      else:
        os.environ['TF_CONFIG'] = json.dumps({
          'cluster': {
              'worker': [node0, node1]
          },
          'task': {'type': 'worker', 'index': 1}
        })
    else:
      os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'evaluator': ["localhost:6060"]
        },
        'task': {'type': 'evaluator', 'index': 0}
      })
  
  
  if FLAGS.use_async_checkpointing:
    save_checkpoints_steps = None
  else:
    save_checkpoints_steps = max(100, FLAGS.iterations_per_loop)
    
  gconfig = tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True)))

  if FLAGS.use_xla:
    gconfig.session_config.graph_options.optimizer_options.global_jit_level = (tf.OptimizerOptions.ON_1)
  
  # mnasnet opt - check if this is required!
  gconfig.gpu_options.allow_growth = True
  #gconfig.session_config.gpu_options.visible_device_list = str(hvd.local_rank())
  
  config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      log_step_count_steps=FLAGS.log_step_count_steps,
      train_distribute=strategy,
      session_config=gconfig)  # pylint: disable=line-too-long

  print('mnasnet opt - config cluster spec', config.cluster_spec)
  
  # Initializes model parameters.
  params = dict(
      steps_per_epoch=FLAGS.num_train_images / FLAGS.train_batch_size,
      batch_size=FLAGS.train_batch_size,
      dtype = tf.float32,
      use_bfloat16=FLAGS.use_bfloat16,
      quantized_training=FLAGS.quantized_training)
  
  mnasnet_est = tf.estimator.Estimator(
      model_fn=mnasnet_model_fn,
      model_dir=FLAGS.model_dir,
      config=config,
      params=params)
  
  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  if FLAGS.bigtable_instance:
    tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
    select_train, select_eval = _select_tables_from_flags()
    imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
        is_training=is_training,
        use_bfloat16=False,
        transpose_input=FLAGS.transpose_input,
        selection=selection) for (is_training, selection) in
                                     [(True, select_train),
                                      (False, select_eval)]]
  else:
    if FLAGS.data_dir == FAKE_DATA_DIR:
      tf.logging.info('Using fake dataset.')
    else:
      tf.logging.info('Using dataset: %s', FLAGS.data_dir)
    imagenet_train, imagenet_eval = [
        imagenet_input.ImageNetInput(
            is_training=is_training,
            data_dir=FLAGS.data_dir,
            transpose_input=FLAGS.transpose_input,
            cache=FLAGS.use_cache and is_training,
            image_size=FLAGS.input_image_size,
            num_parallel_calls=FLAGS.num_parallel_calls,
            use_bfloat16=FLAGS.use_bfloat16) for is_training in [True, False]
    ]

  if FLAGS.mode == 'eval':
    eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size
    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = mnasnet_est.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d', eval_results,
                        elapsed_time)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info('Evaluation finished after training step %d',
                          current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint',
                        ckpt)

    if FLAGS.export_dir:
      export(mnasnet_est, FLAGS.export_dir, FLAGS.post_quantize)
  else:  # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    current_step = estimator._load_global_step_from_checkpoint_dir(  # pylint: disable=protected-access
        FLAGS.model_dir)

    tf.logging.info(
        'Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', FLAGS.train_steps,
        FLAGS.train_steps / params['steps_per_epoch'], current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      hooks = []
      if FLAGS.use_async_checkpointing:
        hooks.append(
            async_checkpoint.AsyncCheckpointSaverHook(
                checkpoint_dir=FLAGS.model_dir,
                save_steps=max(100, FLAGS.iterations_per_loop)))
      mnasnet_est.train(
          input_fn=imagenet_train.input_fn,
          max_steps=FLAGS.train_steps,
          hooks=hooks)

    else:
      assert FLAGS.mode == 'train_and_eval'
      train_spec = tf.estimator.TrainSpec(input_fn=imagenet_train.input_fn, max_steps=FLAGS.train_steps)
      eval_spec = tf.estimator.EvalSpec(input_fn=imagenet_eval.input_fn, steps=FLAGS.num_eval_images // FLAGS.eval_batch_size, throttle_secs=600)
      tf.estimator.train_and_evaluate(mnasnet_est, train_spec, eval_spec)
      
      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                      FLAGS.train_steps, elapsed_time)
      if FLAGS.export_dir:
        export(mnasnet_est, FLAGS.export_dir, FLAGS.post_quantize)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
