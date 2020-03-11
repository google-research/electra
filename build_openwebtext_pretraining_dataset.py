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

"""Preprocessess the Open WebText corpus for ELECTRA pre-training."""

import argparse
import multiprocessing
import os
import random
import tarfile
import time
import tensorflow.compat.v1 as tf

import build_pretraining_dataset
from util import utils


def write_examples(job_id, args):
  """A single process creating and writing out pre-processed examples."""
  job_tmp_dir = os.path.join(args.data_dir, "tmp", "job_" + str(job_id))
  owt_dir = os.path.join(args.data_dir, "openwebtext")

  def log(*args):
    msg = " ".join(map(str, args))
    print("Job {}:".format(job_id), msg)

  log("Creating example writer")
  example_writer = build_pretraining_dataset.ExampleWriter(
      job_id=job_id,
      vocab_file=os.path.join(args.data_dir, "vocab.txt"),
      output_dir=os.path.join(args.data_dir, "pretrain_tfrecords"),
      max_seq_length=args.max_seq_length,
      num_jobs=args.num_processes,
      blanks_separate_docs=False,
      do_lower_case=args.do_lower_case
  )
  log("Writing tf examples")
  fnames = sorted(tf.io.gfile.listdir(owt_dir))
  fnames = [f for (i, f) in enumerate(fnames)
            if i % args.num_processes == job_id]
  random.shuffle(fnames)
  start_time = time.time()
  for file_no, fname in enumerate(fnames):
    if file_no > 0 and file_no % 10 == 0:
      elapsed = time.time() - start_time
      log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, "
          "{:} examples written".format(
              file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
              int((len(fnames) - file_no) / (file_no / elapsed)),
              example_writer.n_written))
    utils.rmkdir(job_tmp_dir)
    with tarfile.open(os.path.join(owt_dir, fname)) as f:
      f.extractall(job_tmp_dir)
    extracted_files = tf.io.gfile.listdir(job_tmp_dir)
    random.shuffle(extracted_files)
    for txt_fname in extracted_files:
      example_writer.write_examples(os.path.join(job_tmp_dir, txt_fname))
  example_writer.finish()
  log("Done!")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", required=True,
                      help="Location of data (vocab file, corpus, etc).")
  parser.add_argument("--max-seq-length", default=128, type=int,
                      help="Number of tokens per example.")
  parser.add_argument("--num-processes", default=1, type=int,
                      help="Parallelize across multiple processes.")
  parser.add_argument("--do-lower-case", dest='do_lower_case',
                      action='store_true', help="Lower case input text.")
  parser.add_argument("--no-lower-case", dest='do_lower_case',
                      action='store_false', help="Don't lower case input text.")
  parser.set_defaults(do_lower_case=True)
  args = parser.parse_args()

  utils.rmkdir(os.path.join(args.data_dir, "pretrain_tfrecords"))
  if args.num_processes == 1:
    write_examples(0, args)
  else:
    jobs = []
    for i in range(args.num_processes):
      job = multiprocessing.Process(target=write_examples, args=(i, args))
      jobs.append(job)
      job.start()
    for job in jobs:
      job.join()


if __name__ == "__main__":
  main()
