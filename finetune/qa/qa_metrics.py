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

"""Evaluation metrics for question-answering tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import six

import configure_finetuning
from finetune import scorer
from finetune.qa import mrqa_official_eval
from finetune.qa import squad_official_eval
from finetune.qa import squad_official_eval_v1
from model import tokenization
from util import utils


RawResult = collections.namedtuple("RawResult", [
    "unique_id", "start_logits", "end_logits", "answerable_logit",
    "start_top_log_probs", "start_top_index", "end_top_log_probs",
    "end_top_index"
])


class SpanBasedQAScorer(scorer.Scorer):
  """Runs evaluation for SQuAD 1.1, SQuAD 2.0, and MRQA tasks."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, task, split,
               v2):
    super(SpanBasedQAScorer, self).__init__()
    self._config = config
    self._task = task
    self._name = task.name
    self._split = split
    self._v2 = v2
    self._all_results = []
    self._total_loss = 0
    self._split = split
    self._eval_examples = task.get_examples(split)

  def update(self, results):
    super(SpanBasedQAScorer, self).update(results)
    self._all_results.append(
        RawResult(
            unique_id=results["eid"],
            start_logits=results["start_logits"],
            end_logits=results["end_logits"],
            answerable_logit=results["answerable_logit"],
            start_top_log_probs=results["start_top_log_probs"],
            start_top_index=results["start_top_index"],
            end_top_log_probs=results["end_top_log_probs"],
            end_top_index=results["end_top_index"],
        ))
    self._total_loss += results["loss"]

  def get_loss(self):
    return self._total_loss / len(self._all_results)

  def _get_results(self):
    self.write_predictions()
    if self._name == "squad":
      squad_official_eval.set_opts(self._config, self._split)
      squad_official_eval.main()
      return sorted(utils.load_json(
          self._config.qa_eval_file(self._name)).items())
    elif self._name == "squadv1":
      return sorted(squad_official_eval_v1.main(
          self._config, self._split).items())
    else:
      return sorted(mrqa_official_eval.main(
          self._config, self._split, self._name).items())

  def write_predictions(self):
    """Write final predictions to the json file."""
    unique_id_to_result = {}
    for result in self._all_results:
      unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit",
         "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for example in self._eval_examples:
      example_id = example.qas_id if "squad" in self._name else example.qid
      features = self._task.featurize(example, False, for_eval=True)

      prelim_predictions = []
      # keep track of the minimum score of null start+end of position 0
      score_null = 1000000  # large and positive
      for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature[self._name + "_eid"]]
        if self._config.joint_prediction:
          start_indexes = result.start_top_index
          end_indexes = result.end_top_index
        else:
          start_indexes = _get_best_indexes(result.start_logits,
                                            self._config.n_best_size)
          end_indexes = _get_best_indexes(result.end_logits,
                                          self._config.n_best_size)
        # if we could have irrelevant answers, get the min score of irrelevant
        if self._v2:
          if self._config.answerable_classifier:
            feature_null_score = result.answerable_logit
          else:
            feature_null_score = result.start_logits[0] + result.end_logits[0]
          if feature_null_score < score_null:
            score_null = feature_null_score
        for i, start_index in enumerate(start_indexes):
          for j, end_index in enumerate(
              end_indexes[i] if self._config.joint_prediction else end_indexes):
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(feature[self._name + "_tokens"]):
              continue
            if end_index >= len(feature[self._name + "_tokens"]):
              continue
            if start_index == 0:
              continue
            if start_index not in feature[self._name + "_token_to_orig_map"]:
              continue
            if end_index not in feature[self._name + "_token_to_orig_map"]:
              continue
            if not feature[self._name + "_token_is_max_context"].get(
                start_index, False):
              continue
            if end_index < start_index:
              continue
            length = end_index - start_index + 1
            if length > self._config.max_answer_length:
              continue
            start_logit = (result.start_top_log_probs[i] if
                           self._config.joint_prediction else
                           result.start_logits[start_index])
            end_logit = (result.end_top_log_probs[i, j] if
                         self._config.joint_prediction else
                         result.end_logits[end_index])
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logit,
                    end_logit=end_logit))

      if self._v2:
        if len(prelim_predictions) == 0 and self._config.debug:
          tokid = sorted(feature[self._name + "_token_to_orig_map"].keys())[0]
          prelim_predictions.append(_PrelimPrediction(
              feature_index=0,
              start_index=tokid,
              end_index=tokid + 1,
              start_logit=1.0,
              end_logit=1.0))
      prelim_predictions = sorted(
          prelim_predictions,
          key=lambda x: (x.start_logit + x.end_logit),
          reverse=True)

      _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
          "NbestPrediction", ["text", "start_logit", "end_logit"])

      seen_predictions = {}
      nbest = []
      for pred in prelim_predictions:
        if len(nbest) >= self._config.n_best_size:
          break
        feature = features[pred.feature_index]
        tok_tokens = feature[self._name + "_tokens"][
            pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature[
            self._name + "_token_to_orig_map"][pred.start_index]
        orig_doc_end = feature[
            self._name + "_token_to_orig_map"][pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(self._config, tok_text, orig_text)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))

      # In very rare edge cases we could have no valid predictions. So we
      # just create a nonce prediction in this case to avoid failure.
      if not nbest:
        nbest.append(
            _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

      assert len(nbest) >= 1

      total_scores = []
      best_non_null_entry = None
      for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
          if entry.text:
            best_non_null_entry = entry

      probs = _compute_softmax(total_scores)

      nbest_json = []
      for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(dict(output))

      assert len(nbest_json) >= 1

      if not self._v2:
        all_predictions[example_id] = nbest_json[0]["text"]
      else:
        # predict "" iff the null score - the score of best non-null > threshold
        if self._config.answerable_classifier:
          score_diff = score_null
        else:
          score_diff = score_null - best_non_null_entry.start_logit - (
              best_non_null_entry.end_logit)
        scores_diff_json[example_id] = score_diff
        all_predictions[example_id] = best_non_null_entry.text

      all_nbest_json[example_id] = nbest_json

    utils.write_json(dict(all_predictions),
                     self._config.qa_preds_file(self._name))
    if self._v2:
      utils.write_json({
          k: float(v) for k, v in six.iteritems(scores_diff_json)},
          self._config.qa_na_file(self._name))


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = np.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def get_final_text(config: configure_finetuning.FinetuningConfig, pred_text,
                   orig_text):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for i, c in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return ns_text, dict(ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=config.do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if config.debug:
      utils.log(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if config.debug:
      utils.log("Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if config.debug:
      utils.log("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if config.debug:
      utils.log("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text
