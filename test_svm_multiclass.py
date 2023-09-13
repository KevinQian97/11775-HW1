#!/bin/python

import argparse
import numpy as np
import os
from sklearn.svm.classes import SVC
import pickle
import sys
import numpy as np

# Apply the SVM model to the testing videos;
# Output the prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()


  feat_list = []
  video_ids = []
  for line in open(args.list_videos).readlines()[1:]:
    video_id= line.strip().split(",")[0]
    video_ids.append(video_id)
    # pdb.set_trace()
    #feat_filepath = os.path.join(args.feat_dir, video_id + ".kmeans.csv")
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype='float'))

  # Load model and get scores with trained svm model and feat_list
  # the shape of scoress should be (num_samples, num_class)




  scoress = 
  raise NotImplementedError("Please fill in the blank first")
  
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, scores in enumerate(scoress):
      predicted_class = np.argmax(scores)
      f.writelines("%s,%d\n" % (video_ids[i], predicted_class))
