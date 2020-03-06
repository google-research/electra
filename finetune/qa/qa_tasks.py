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

"""Question answering tasks. SQuAD 1.1/2.0 and 2019 MRQA tasks are supported."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import json
import os
import six
import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import feature_spec
from finetune import task
from finetune.qa import qa_metrics
from model import modeling
from model import tokenization
from util import utils


class QAExample(task.Example):
  """Question-answering example."""

  def __init__(self,
               task_name,
               eid,
               qas_id,
               qid,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    super(QAExample, self).__init__(task_name)
    self.eid = eid
    self.qas_id = qas_id
    self.qid = qid
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % self.start_position
    if self.start_position:
      s += ", end_position: %d" % self.end_position
    if self.start_position:
      s += ", is_impossible: %r" % self.is_impossible
    return s


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return new_start, new_end

  return input_start, input_end


def is_whitespace(c):
  return c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F


class QATask(task.Task):
  """A span-based question answering tasks (e.g., SQuAD)."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer, v2=False):
    super(QATask, self).__init__(config, name)
    self._tokenizer = tokenizer
    self._examples = {}
    self.v2 = v2

  def _add_examples(self, examples, example_failures, paragraph, split):
    paragraph_text = paragraph["context"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
      if is_whitespace(c):
        prev_is_whitespace = True
      else:
        if prev_is_whitespace:
          doc_tokens.append(c)
        else:
          doc_tokens[-1] += c
        prev_is_whitespace = False
      char_to_word_offset.append(len(doc_tokens) - 1)

    for qa in paragraph["qas"]:
      qas_id = qa["id"] if "id" in qa else None
      qid = qa["qid"] if "qid" in qa else None
      question_text = qa["question"]
      start_position = None
      end_position = None
      orig_answer_text = None
      is_impossible = False
      if split == "train":
        if self.v2:
          is_impossible = qa["is_impossible"]
        if not is_impossible:
          if "detected_answers" in qa:  # MRQA format
            answer = qa["detected_answers"][0]
            answer_offset = answer["char_spans"][0][0]
          else:  # SQuAD format
            answer = qa["answers"][0]
            answer_offset = answer["answer_start"]
          orig_answer_text = answer["text"]
          answer_length = len(orig_answer_text)
          start_position = char_to_word_offset[answer_offset]
          if answer_offset + answer_length - 1 >= len(char_to_word_offset):
            utils.log("End position is out of document!")
            example_failures[0] += 1
            continue
          end_position = char_to_word_offset[answer_offset + answer_length - 1]

          # Only add answers where the text can be exactly recovered from the
          # document. If this CAN'T happen it's likely due to weird Unicode
          # stuff so we will just skip the example.
          #
          # Note that this means for training mode, every example is NOT
          # guaranteed to be preserved.
          actual_text = " ".join(
              doc_tokens[start_position:(end_position + 1)])
          cleaned_answer_text = " ".join(
              tokenization.whitespace_tokenize(orig_answer_text))
          actual_text = actual_text.lower()
          cleaned_answer_text = cleaned_answer_text.lower()
          if actual_text.find(cleaned_answer_text) == -1:
            utils.log("Could not find answer: '{:}' in doc vs. "
                      "'{:}' in provided answer".format(
                          tokenization.printable_text(actual_text),
                          tokenization.printable_text(cleaned_answer_text)))
            example_failures[0] += 1
            continue
        else:
          start_position = -1
          end_position = -1
          orig_answer_text = ""

      example = QAExample(
          task_name=self.name,
          eid=len(examples),
          qas_id=qas_id,
          qid=qid,
          question_text=question_text,
          doc_tokens=doc_tokens,
          orig_answer_text=orig_answer_text,
          start_position=start_position,
          end_position=end_position,
          is_impossible=is_impossible)
      examples.append(example)

  def get_feature_specs(self):
    return [
        feature_spec.FeatureSpec(self.name + "_eid", []),
        feature_spec.FeatureSpec(self.name + "_start_positions", []),
        feature_spec.FeatureSpec(self.name + "_end_positions", []),
        feature_spec.FeatureSpec(self.name + "_is_impossible", []),
    ]

  def featurize(self, example: QAExample, is_training, log=False,
                for_eval=False):
    all_features = []
    query_tokens = self._tokenizer.tokenize(example.question_text)

    if len(query_tokens) > self.config.max_query_length:
      query_tokens = query_tokens[0:self.config.max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = self._tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1
    if is_training and not example.is_impossible:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, self._tokenizer,
          example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = self.config.max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, self.config.doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < self.config.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == self.config.max_seq_length
      assert len(input_mask) == self.config.max_seq_length
      assert len(segment_ids) == self.config.max_seq_length

      start_position = None
      end_position = None
      if is_training and not example.is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and example.is_impossible:
        start_position = 0
        end_position = 0

      if log:
        utils.log("*** Example ***")
        utils.log("doc_span_index: %s" % doc_span_index)
        utils.log("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        utils.log("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        utils.log("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        utils.log("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        utils.log("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        utils.log("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if is_training and example.is_impossible:
          utils.log("impossible example")
        if is_training and not example.is_impossible:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          utils.log("start_position: %d" % start_position)
          utils.log("end_position: %d" % end_position)
          utils.log("answer: %s" % (tokenization.printable_text(answer_text)))

      features = {
          "task_id": self.config.task_names.index(self.name),
          self.name + "_eid": (1000 * example.eid) + doc_span_index,
          "input_ids": input_ids,
          "input_mask": input_mask,
          "segment_ids": segment_ids,
      }
      if for_eval:
        features.update({
            self.name + "_doc_span_index": doc_span_index,
            self.name + "_tokens": tokens,
            self.name + "_token_to_orig_map": token_to_orig_map,
            self.name + "_token_is_max_context": token_is_max_context,
        })
      if is_training:
        features.update({
            self.name + "_start_positions": start_position,
            self.name + "_end_positions": end_position,
            self.name + "_is_impossible": example.is_impossible
        })
      all_features.append(features)
    return all_features

  def get_prediction_module(self, bert_model, features, is_training,
                            percent_done):
    final_hidden = bert_model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]

    answer_mask = tf.cast(features["input_mask"], tf.float32)
    answer_mask *= tf.cast(features["segment_ids"], tf.float32)
    answer_mask += tf.one_hot(0, seq_length)

    start_logits = tf.squeeze(tf.layers.dense(final_hidden, 1), -1)

    start_top_log_probs = tf.zeros([batch_size, self.config.beam_size])
    start_top_index = tf.zeros([batch_size, self.config.beam_size], tf.int32)
    end_top_log_probs = tf.zeros([batch_size, self.config.beam_size,
                                  self.config.beam_size])
    end_top_index = tf.zeros([batch_size, self.config.beam_size,
                              self.config.beam_size], tf.int32)
    if self.config.joint_prediction:
      start_logits += 1000.0 * (answer_mask - 1)
      start_log_probs = tf.nn.log_softmax(start_logits)
      start_top_log_probs, start_top_index = tf.nn.top_k(
          start_log_probs, k=self.config.beam_size)

      if not is_training:
        # batch, beam, length, hidden
        end_features = tf.tile(tf.expand_dims(final_hidden, 1),
                               [1, self.config.beam_size, 1, 1])
        # batch, beam, length
        start_index = tf.one_hot(start_top_index,
                                 depth=seq_length, axis=-1, dtype=tf.float32)
        # batch, beam, hidden
        start_features = tf.reduce_sum(
            tf.expand_dims(final_hidden, 1) *
            tf.expand_dims(start_index, -1), axis=-2)
        # batch, beam, length, hidden
        start_features = tf.tile(tf.expand_dims(start_features, 2),
                                 [1, 1, seq_length, 1])
      else:
        start_index = tf.one_hot(
            features[self.name + "_start_positions"], depth=seq_length,
            axis=-1, dtype=tf.float32)
        start_features = tf.reduce_sum(tf.expand_dims(start_index, -1) *
                                       final_hidden, axis=1)
        start_features = tf.tile(tf.expand_dims(start_features, 1),
                                 [1, seq_length, 1])
        end_features = final_hidden

      final_repr = tf.concat([start_features, end_features], -1)
      final_repr = tf.layers.dense(final_repr, 512, activation=modeling.gelu,
                                   name="qa_hidden")
      # batch, beam, length (batch, length when training)
      end_logits = tf.squeeze(tf.layers.dense(final_repr, 1), -1,
                              name="qa_logits")
      if is_training:
        end_logits += 1000.0 * (answer_mask - 1)
      else:
        end_logits += tf.expand_dims(1000.0 * (answer_mask - 1), 1)

      if not is_training:
        end_log_probs = tf.nn.log_softmax(end_logits)
        end_top_log_probs, end_top_index = tf.nn.top_k(
            end_log_probs, k=self.config.beam_size)
        end_logits = tf.zeros([batch_size, seq_length])
    else:
      end_logits = tf.squeeze(tf.layers.dense(final_hidden, 1), -1)
      start_logits += 1000.0 * (answer_mask - 1)
      end_logits += 1000.0 * (answer_mask - 1)

    def compute_loss(logits, positions):
      one_hot_positions = tf.one_hot(
          positions, depth=seq_length, dtype=tf.float32)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      loss = -tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
      return loss

    start_positions = features[self.name + "_start_positions"]
    end_positions = features[self.name + "_end_positions"]

    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)

    losses = (start_loss + end_loss) / 2.0

    answerable_logit = tf.zeros([batch_size])
    if self.config.answerable_classifier:
      final_repr = final_hidden[:, 0]
      if self.config.answerable_uses_start_logits:
        start_p = tf.nn.softmax(start_logits)
        start_feature = tf.reduce_sum(tf.expand_dims(start_p, -1) *
                                      final_hidden, axis=1)
        final_repr = tf.concat([final_repr, start_feature], -1)
        final_repr = tf.layers.dense(final_repr, 512,
                                     activation=modeling.gelu)
      answerable_logit = tf.squeeze(tf.layers.dense(final_repr, 1), -1)
      answerable_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.cast(features[self.name + "_is_impossible"], tf.float32),
          logits=answerable_logit)
      losses += answerable_loss * self.config.answerable_weight

    return losses, dict(
        loss=losses,
        start_logits=start_logits,
        end_logits=end_logits,
        answerable_logit=answerable_logit,
        start_positions=features[self.name + "_start_positions"],
        end_positions=features[self.name + "_end_positions"],
        start_top_log_probs=start_top_log_probs,
        start_top_index=start_top_index,
        end_top_log_probs=end_top_log_probs,
        end_top_index=end_top_index,
        eid=features[self.name + "_eid"],
    )

  def get_scorer(self, split="dev"):
    return qa_metrics.SpanBasedQAScorer(self.config, self, split, self.v2)


class MRQATask(QATask):
  """Class for finetuning tasks from the 2019 MRQA shared task."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer):
    super(MRQATask, self).__init__(config, name, tokenizer)

  def get_examples(self, split):
    if split in self._examples:
      utils.log("N EXAMPLES", split, len(self._examples[split]))
      return self._examples[split]

    examples = []
    example_failures = [0]
    with tf.io.gfile.GFile(os.path.join(
        self.config.raw_data_dir(self.name), split + ".jsonl"), "r") as f:
      for i, line in enumerate(f):
        if self.config.debug and i > 10:
          break
        paragraph = json.loads(line.strip())
        if "header" in paragraph:
          continue
        self._add_examples(examples, example_failures, paragraph, split)
    self._examples[split] = examples
    utils.log("{:} examples created, {:} failures".format(
        len(examples), example_failures[0]))
    return examples

  def get_scorer(self, split="dev"):
    return qa_metrics.SpanBasedQAScorer(self.config, self, split, self.v2)


class SQuADTask(QATask):
  """Class for finetuning on SQuAD 2.0 or 1.1."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer, v2=False):
    super(SQuADTask, self).__init__(config, name, tokenizer, v2=v2)

  def get_examples(self, split):
    if split in self._examples:
      return self._examples[split]

    with tf.io.gfile.GFile(os.path.join(
        self.config.raw_data_dir(self.name),
        split + ("-debug" if self.config.debug else "") + ".json"), "r") as f:
      input_data = json.load(f)["data"]

    examples = []
    example_failures = [0]
    for entry in input_data:
      for paragraph in entry["paragraphs"]:
        self._add_examples(examples, example_failures, paragraph, split)
    self._examples[split] = examples
    utils.log("{:} examples created, {:} failures".format(
        len(examples), example_failures[0]))
    return examples

  def get_scorer(self, split="dev"):
    return qa_metrics.SpanBasedQAScorer(self.config, self, split, self.v2)


class SQuAD(SQuADTask):
  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(SQuAD, self).__init__(config, "squad", tokenizer, v2=True)


class SQuADv1(SQuADTask):
  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(SQuADv1, self).__init__(config, "squadv1", tokenizer)


class NewsQA(MRQATask):
  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(NewsQA, self).__init__(config, "newsqa", tokenizer)


class NaturalQuestions(MRQATask):
  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(NaturalQuestions, self).__init__(config, "naturalqs", tokenizer)


class SearchQA(MRQATask):
  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(SearchQA, self).__init__(config, "searchqa", tokenizer)


class TriviaQA(MRQATask):
  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(TriviaQA, self).__init__(config, "triviaqa", tokenizer)
