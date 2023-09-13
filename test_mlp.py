#!/bin/python

import argparse
from operator import not_
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier

# Apply the MLP model to the testing videos;
# Output prediction class for each video

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_file")
  parser.add_argument("feat_dir")
  parser.add_argument("feat_dim", type=int)
  parser.add_argument("list_videos")
  parser.add_argument("output_file")
  parser.add_argument("--file_ext", default=".csv")

  return parser.parse_args()


if __name__ == '__main__':

  args = parse_args()

  feat_list = []
  video_ids = []
  not_found_count = 0
  for line in open(args.list_videos).readlines()[1:]:
    video_id= line.strip().split(",")[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.feat_dir, video_id + args.file_ext)
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
      not_found_count += 1
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

  if not_found_count > 0:
    print(f'Could not find the features for {not_found_count} samples.')

  # Load model and get predictions
  # the shape of pred_classes should be (num_samples)




  pred_classes = 
  raise NotImplementedError("please fill in the blank first")

  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
