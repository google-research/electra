# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Helpers for preparing pre-training data and supplying them to the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow.compat.v1 as tf

import configure_pretraining
from model import tokenization
from util import utils


def get_input_fn(config: configure_pretraining.PretrainingConfig, is_training,
                 num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  input_files = []
  for input_pattern in config.pretrain_tfrecords.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
    }

    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.repeat()
    d = d.shuffle(buffer_size=len(input_files))

    # `cycle_length` is the number of parallel files that get read.
    cycle_length = min(num_cpu_threads, len(input_files))

    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    d = d.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=is_training,
            cycle_length=cycle_length))
    d = d.shuffle(buffer_size=100)

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don"t* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example


# model inputs - it's a bit nicer to use a namedtuple rather than keep the
# features as a dict
Inputs = collections.namedtuple(
    "Inputs", ["input_ids", "input_mask", "segment_ids", "masked_lm_positions",
               "masked_lm_ids", "masked_lm_weights"])


def features_to_inputs(features):
  return Inputs(
      input_ids=features["input_ids"],
      input_mask=features["input_mask"],
      segment_ids=features["segment_ids"],
      masked_lm_positions=(features["masked_lm_positions"]
                           if "masked_lm_positions" in features else None),
      masked_lm_ids=(features["masked_lm_ids"]
                     if "masked_lm_ids" in features else None),
      masked_lm_weights=(features["masked_lm_weights"]
                         if "masked_lm_weights" in features else None),
  )


def get_updated_inputs(inputs, **kwargs):
  features = inputs._asdict()
  for k, v in kwargs.items():
    features[k] = v
  return features_to_inputs(features)


ENDC = "\033[0m"
COLORS = ["\033[" + str(n) + "m" for n in list(range(91, 97)) + [90]]
RED = COLORS[0]
BLUE = COLORS[3]
CYAN = COLORS[5]
GREEN = COLORS[1]


def print_tokens(inputs: Inputs, inv_vocab, updates_mask=None):
  """Pretty-print model inputs."""
  pos_to_tokid = {}
  for tokid, pos, weight in zip(
      inputs.masked_lm_ids[0], inputs.masked_lm_positions[0],
      inputs.masked_lm_weights[0]):
    if weight == 0:
      pass
    else:
      pos_to_tokid[pos] = tokid

  text = ""
  provided_update_mask = (updates_mask is not None)
  if not provided_update_mask:
    updates_mask = np.zeros_like(inputs.input_ids)
  for pos, (tokid, um) in enumerate(
      zip(inputs.input_ids[0], updates_mask[0])):
    token = inv_vocab[tokid]
    if token == "[PAD]":
      break
    if pos in pos_to_tokid:
      token = RED + token + " (" + inv_vocab[pos_to_tokid[pos]] + ")" + ENDC
      if provided_update_mask:
        assert um == 1
    else:
      if provided_update_mask:
        assert um == 0
    text += token + " "
  utils.log(tokenization.printable_text(text))
