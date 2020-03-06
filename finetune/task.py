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

"""Defines a supervised NLP task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import List, Tuple

import configure_finetuning
from finetune import feature_spec
from finetune import scorer
from model import modeling


class Example(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, task_name):
    self.task_name = task_name


class Task(object):
  """Override this class to add a new fine-tuning task."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name):
    self.config = config
    self.name = name

  def get_test_splits(self):
    return ["test"]

  @abc.abstractmethod
  def get_examples(self, split):
    pass

  @abc.abstractmethod
  def get_scorer(self) -> scorer.Scorer:
    pass

  @abc.abstractmethod
  def get_feature_specs(self) -> List[feature_spec.FeatureSpec]:
    pass

  @abc.abstractmethod
  def featurize(self, example: Example, is_training: bool,
                log: bool=False):
    pass

  @abc.abstractmethod
  def get_prediction_module(
      self, bert_model: modeling.BertModel, features: dict, is_training: bool,
      percent_done: float) -> Tuple:
    pass

  def __repr__(self):
    return "Task(" + self.name + ")"
