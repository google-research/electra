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

"""Helper functions for pre-training. These mainly deal with the gathering and
scattering needed so the generator only makes predictions for the small number
of masked tokens.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

import configure_pretraining
from model import modeling
from model import tokenization
from pretrain import pretrain_data


def gather_positions(sequence, positions):
  """Gathers the vectors at the specific positions over a minibatch.

  Args:
    sequence: A [batch_size, seq_length] or
        [batch_size, seq_length, depth] tensor of values
    positions: A [batch_size, n_positions] tensor of indices

  Returns: A [batch_size, n_positions] or
    [batch_size, n_positions, depth] tensor of the values at the indices
  """
  shape = modeling.get_shape_list(sequence, expected_rank=[2, 3])
  depth_dimension = (len(shape) == 3)
  if depth_dimension:
    B, L, D = shape
  else:
    B, L = shape
    D = 1
    sequence = tf.expand_dims(sequence, -1)
  position_shift = tf.expand_dims(L * tf.range(B), -1)
  flat_positions = tf.reshape(positions + position_shift, [-1])
  flat_sequence = tf.reshape(sequence, [B * L, D])
  gathered = tf.gather(flat_sequence, flat_positions)
  if depth_dimension:
    return tf.reshape(gathered, [B, -1, D])
  else:
    return tf.reshape(gathered, [B, -1])


def scatter_update(sequence, updates, positions):
  """Scatter-update a sequence.

  Args:
    sequence: A [batch_size, seq_len] or [batch_size, seq_len, depth] tensor
    updates: A tensor of size batch_size*seq_len(*depth)
    positions: A [batch_size, n_positions] tensor

  Returns: A tuple of two tensors. First is a [batch_size, seq_len] or
    [batch_size, seq_len, depth] tensor of "sequence" with elements at
    "positions" replaced by the values at "updates." Updates to index 0 are
    ignored. If there are duplicated positions the update is only applied once.
    Second is a [batch_size, seq_len] mask tensor of which inputs were updated.
  """
  shape = modeling.get_shape_list(sequence, expected_rank=[2, 3])
  depth_dimension = (len(shape) == 3)
  if depth_dimension:
    B, L, D = shape
  else:
    B, L = shape
    D = 1
    sequence = tf.expand_dims(sequence, -1)
  N = modeling.get_shape_list(positions)[1]

  shift = tf.expand_dims(L * tf.range(B), -1)
  flat_positions = tf.reshape(positions + shift, [-1, 1])
  flat_updates = tf.reshape(updates, [-1, D])
  updates = tf.scatter_nd(flat_positions, flat_updates, [B * L, D])
  updates = tf.reshape(updates, [B, L, D])

  flat_updates_mask = tf.ones([B * N], tf.int32)
  updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask, [B * L])
  updates_mask = tf.reshape(updates_mask, [B, L])
  not_first_token = tf.concat([tf.zeros((B, 1), tf.int32),
                               tf.ones((B, L - 1), tf.int32)], -1)
  updates_mask *= not_first_token
  updates_mask_3d = tf.expand_dims(updates_mask, -1)

  # account for duplicate positions
  if sequence.dtype == tf.float32:
    updates_mask_3d = tf.cast(updates_mask_3d, tf.float32)
    updates /= tf.maximum(1.0, updates_mask_3d)
  else:
    assert sequence.dtype == tf.int32
    updates = tf.math.floordiv(updates, tf.maximum(1, updates_mask_3d))
  updates_mask = tf.minimum(updates_mask, 1)
  updates_mask_3d = tf.minimum(updates_mask_3d, 1)

  updated_sequence = (((1 - updates_mask_3d) * sequence) +
                      (updates_mask_3d * updates))
  if not depth_dimension:
    updated_sequence = tf.squeeze(updated_sequence, -1)

  return updated_sequence, updates_mask


def _get_candidates_mask(inputs: pretrain_data.Inputs, vocab,
                         disallow_from_mask=None):
  """Returns a mask tensor of positions in the input that can be masked out."""
  ignore_ids = [vocab["[SEP]"], vocab["[CLS]"], vocab["[MASK]"]]
  candidates_mask = tf.ones_like(inputs.input_ids, tf.bool)
  for ignore_id in ignore_ids:
    candidates_mask &= tf.not_equal(inputs.input_ids, ignore_id)
  candidates_mask &= tf.cast(inputs.input_mask, tf.bool)
  if disallow_from_mask is not None:
    candidates_mask &= ~disallow_from_mask
  return candidates_mask


def mask(config: configure_pretraining.PretrainingConfig,
         inputs: pretrain_data.Inputs, mask_prob, proposal_distribution=1.0,
         disallow_from_mask=None, already_masked=None):
  """Implementation of dynamic masking. The optional arguments aren't needed for
  BERT/ELECTRA and are from early experiments in "strategically" masking out
  tokens instead of uniformly at random.

  Args:
    config: configure_pretraining.PretrainingConfig
    inputs: pretrain_data.Inputs containing input input_ids/input_mask
    mask_prob: percent of tokens to mask
    proposal_distribution: for non-uniform masking can be a [B, L] tensor
                           of scores for masking each position.
    disallow_from_mask: a boolean tensor of [B, L] of positions that should
                        not be masked out
    already_masked: a boolean tensor of [B, N] of already masked-out tokens
                    for multiple rounds of masking
  Returns: a pretrain_data.Inputs with masking added
  """
  # Get the batch size, sequence length, and max masked-out tokens
  N = config.max_predictions_per_seq
  B, L = modeling.get_shape_list(inputs.input_ids)

  # Find indices where masking out a token is allowed
  vocab = tokenization.FullTokenizer(
      config.vocab_file, do_lower_case=config.do_lower_case).vocab
  candidates_mask = _get_candidates_mask(inputs, vocab, disallow_from_mask)

  # Set the number of tokens to mask out per example
  num_tokens = tf.cast(tf.reduce_sum(inputs.input_mask, -1), tf.float32)
  num_to_predict = tf.maximum(1, tf.minimum(
      N, tf.cast(tf.round(num_tokens * mask_prob), tf.int32)))
  masked_lm_weights = tf.cast(tf.sequence_mask(num_to_predict, N), tf.float32)
  if already_masked is not None:
    masked_lm_weights *= (1 - already_masked)

  # Get a probability of masking each position in the sequence
  candidate_mask_float = tf.cast(candidates_mask, tf.float32)
  sample_prob = (proposal_distribution * candidate_mask_float)
  sample_prob /= tf.reduce_sum(sample_prob, axis=-1, keepdims=True)

  # Sample the positions to mask out
  sample_prob = tf.stop_gradient(sample_prob)
  sample_logits = tf.log(sample_prob)
  masked_lm_positions = tf.random.categorical(
      sample_logits, N, dtype=tf.int32)
  masked_lm_positions *= tf.cast(masked_lm_weights, tf.int32)

  # Get the ids of the masked-out tokens
  shift = tf.expand_dims(L * tf.range(B), -1)
  flat_positions = tf.reshape(masked_lm_positions + shift, [-1, 1])
  masked_lm_ids = tf.gather_nd(tf.reshape(inputs.input_ids, [-1]),
                               flat_positions)
  masked_lm_ids = tf.reshape(masked_lm_ids, [B, -1])
  masked_lm_ids *= tf.cast(masked_lm_weights, tf.int32)

  # Update the input ids
  replace_with_mask_positions = masked_lm_positions * tf.cast(
      tf.less(tf.random.uniform([B, N]), 0.85), tf.int32)
  inputs_ids, _ = scatter_update(
      inputs.input_ids, tf.fill([B, N], vocab["[MASK]"]),
      replace_with_mask_positions)

  return pretrain_data.get_updated_inputs(
      inputs,
      input_ids=tf.stop_gradient(inputs_ids),
      masked_lm_positions=masked_lm_positions,
      masked_lm_ids=masked_lm_ids,
      masked_lm_weights=masked_lm_weights
  )


def unmask(inputs: pretrain_data.Inputs):
  unmasked_input_ids, _ = scatter_update(
      inputs.input_ids, inputs.masked_lm_ids, inputs.masked_lm_positions)
  return pretrain_data.get_updated_inputs(inputs, input_ids=unmasked_input_ids)


def sample_from_softmax(logits, disallow=None):
  if disallow is not None:
    logits -= 1000.0 * disallow
  uniform_noise = tf.random.uniform(
      modeling.get_shape_list(logits), minval=0, maxval=1)
  gumbel_noise = -tf.log(-tf.log(uniform_noise + 1e-9) + 1e-9)
  return tf.one_hot(tf.argmax(tf.nn.softmax(logits + gumbel_noise), -1,
                              output_type=tf.int32), logits.shape[-1])
