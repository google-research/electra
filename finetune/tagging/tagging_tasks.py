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

"""Sequence tagging tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os
import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import feature_spec
from finetune import task
from finetune.tagging import tagging_metrics
from finetune.tagging import tagging_utils
from model import tokenization
from pretrain import pretrain_helpers
from util import utils


LABEL_ENCODING = "BIOES"


class TaggingExample(task.Example):
  """A single tagged input sequence."""

  def __init__(self, eid, task_name, words, tags, is_token_level,
               label_mapping):
    super(TaggingExample, self).__init__(task_name)
    self.eid = eid
    self.words = words
    if is_token_level:
      labels = tags
    else:
      span_labels = tagging_utils.get_span_labels(tags)
      labels = tagging_utils.get_tags(
          span_labels, len(words), LABEL_ENCODING)
    self.labels = [label_mapping[l] for l in labels]


class TaggingTask(task.Task):
  """Defines a sequence tagging task (e.g., part-of-speech tagging)."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer, is_token_level):
    super(TaggingTask, self).__init__(config, name)
    self._tokenizer = tokenizer
    self._label_mapping_path = os.path.join(
        self.config.preprocessed_data_dir,
        ("debug_" if self.config.debug else "") + self.name +
        "_label_mapping.pkl")
    self._is_token_level = is_token_level
    self._label_mapping = None

  def get_examples(self, split):
    sentences = self._get_labeled_sentences(split)
    examples = []
    label_mapping = self._get_label_mapping(split, sentences)
    for i, (words, tags) in enumerate(sentences):
      examples.append(TaggingExample(
          i, self.name, words, tags, self._is_token_level, label_mapping
      ))
    return examples

  def _get_label_mapping(self, provided_split=None, provided_sentences=None):
    if self._label_mapping is not None:
      return self._label_mapping
    if tf.io.gfile.exists(self._label_mapping_path):
      self._label_mapping = utils.load_pickle(self._label_mapping_path)
      return self._label_mapping
    utils.log("Writing label mapping for task", self.name)
    tag_counts = collections.Counter()
    train_tags = set()
    for split in ["train", "dev", "test"]:
      if not tf.io.gfile.exists(os.path.join(
          self.config.raw_data_dir(self.name), split + ".txt")):
        continue
      if split == provided_split:
        split_sentences = provided_sentences
      else:
        split_sentences = self._get_labeled_sentences(split)
      for _, tags in split_sentences:
        if not self._is_token_level:
          span_labels = tagging_utils.get_span_labels(tags)
          tags = tagging_utils.get_tags(span_labels, len(tags), LABEL_ENCODING)
        for tag in tags:
          tag_counts[tag] += 1
          if provided_split == "train":
            train_tags.add(tag)
    if self.name == "ccg":
      infrequent_tags = []
      for tag in tag_counts:
        if tag not in train_tags:
          infrequent_tags.append(tag)
      label_mapping = {
          label: i for i, label in enumerate(sorted(filter(
              lambda t: t not in infrequent_tags, tag_counts.keys())))
      }
      n = len(label_mapping)
      for tag in infrequent_tags:
        label_mapping[tag] = n
    else:
      labels = sorted(tag_counts.keys())
      label_mapping = {label: i for i, label in enumerate(labels)}
    utils.write_pickle(label_mapping, self._label_mapping_path)
    self._label_mapping = label_mapping
    return label_mapping

  def featurize(self, example: TaggingExample, is_training, log=False):
    words_to_tokens = tokenize_and_align(self._tokenizer, example.words)
    input_ids = []
    tagged_positions = []
    for word_tokens in words_to_tokens:
      if len(words_to_tokens) + len(input_ids) + 1 > self.config.max_seq_length:
        input_ids.append(self._tokenizer.vocab["[SEP]"])
        break
      if "[CLS]" not in word_tokens and "[SEP]" not in word_tokens:
        tagged_positions.append(len(input_ids))
      for token in word_tokens:
        input_ids.append(self._tokenizer.vocab[token])

    pad = lambda x: x + [0] * (self.config.max_seq_length - len(x))
    labels = pad(example.labels[:self.config.max_seq_length])
    labeled_positions = pad(tagged_positions)
    labels_mask = pad([1.0] * len(tagged_positions))
    segment_ids = pad([1] * len(input_ids))
    input_mask = pad([1] * len(input_ids))
    input_ids = pad(input_ids)
    assert len(input_ids) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(segment_ids) == self.config.max_seq_length
    assert len(labels) == self.config.max_seq_length
    assert len(labels_mask) == self.config.max_seq_length

    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "task_id": self.config.task_names.index(self.name),
        self.name + "_eid": example.eid,
        self.name + "_labels": labels,
        self.name + "_labels_mask": labels_mask,
        self.name + "_labeled_positions": labeled_positions
    }

  def _get_labeled_sentences(self, split):
    sentences = []
    with tf.io.gfile.GFile(os.path.join(self.config.raw_data_dir(self.name),
                                        split + ".txt"), "r") as f:
      sentence = []
      for line in f:
        line = line.strip().split()
        if not line:
          if sentence:
            words, tags = zip(*sentence)
            sentences.append((words, tags))
            sentence = []
            if self.config.debug and len(sentences) > 100:
              return sentences
          continue
        if line[0] == "-DOCSTART-":
          continue
        word, tag = line[0], line[-1]
        sentence.append((word, tag))
    return sentences

  def get_scorer(self):
    return tagging_metrics.AccuracyScorer() if self._is_token_level else \
      tagging_metrics.EntityLevelF1Scorer(self._get_label_mapping())

  def get_feature_specs(self):
    return [
        feature_spec.FeatureSpec(self.name + "_eid", []),
        feature_spec.FeatureSpec(self.name + "_labels",
                                 [self.config.max_seq_length]),
        feature_spec.FeatureSpec(self.name + "_labels_mask",
                                 [self.config.max_seq_length],
                                 is_int_feature=False),
        feature_spec.FeatureSpec(self.name + "_labeled_positions",
                                 [self.config.max_seq_length]),
    ]

  def get_prediction_module(
      self, bert_model, features, is_training, percent_done):
    n_classes = len(self._get_label_mapping())
    reprs = bert_model.get_sequence_output()
    reprs = pretrain_helpers.gather_positions(
        reprs, features[self.name + "_labeled_positions"])
    logits = tf.layers.dense(reprs, n_classes)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(features[self.name + "_labels"], n_classes),
        logits=logits)
    losses *= features[self.name + "_labels_mask"]
    losses = tf.reduce_sum(losses, axis=-1)
    return losses, dict(
        loss=losses,
        logits=logits,
        predictions=tf.argmax(logits, axis=-1),
        labels=features[self.name + "_labels"],
        labels_mask=features[self.name + "_labels_mask"],
        eid=features[self.name + "_eid"],
    )

  def _create_examples(self, lines, split):
    pass


def tokenize_and_align(tokenizer, words, cased=False):
  """Splits up words into subword-level tokens."""
  words = ["[CLS]"] + list(words) + ["[SEP]"]
  basic_tokenizer = tokenizer.basic_tokenizer
  tokenized_words = []
  for word in words:
    word = tokenization.convert_to_unicode(word)
    word = basic_tokenizer._clean_text(word)
    if word == "[CLS]" or word == "[SEP]":
      word_toks = [word]
    else:
      if not cased:
        word = word.lower()
        word = basic_tokenizer._run_strip_accents(word)
      word_toks = basic_tokenizer._run_split_on_punc(word)
    tokenized_word = []
    for word_tok in word_toks:
      tokenized_word += tokenizer.wordpiece_tokenizer.tokenize(word_tok)
    tokenized_words.append(tokenized_word)
  assert len(tokenized_words) == len(words)
  return tokenized_words


class Chunking(TaggingTask):
  """Text chunking."""

  def __init__(self, config, tokenizer):
    super(Chunking, self).__init__(config, "chunk", tokenizer, False)
