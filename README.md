# Instructions for hw1
In this homework, we will perform a video classification task using audio-only features. Since this is the first and preliminary assignment of this course, you will also learn some basic steps of sklearn, pytorch and GCP.

## Environment Settings
We suggest using [conda](https://docs.conda.io/en/latest/) to manage your packages. You can quickly check or install the required packages from `environment.yaml`.

If you use conda, you should easilly install the packages through:
```
conda env create -f environment.yaml
```

Major Depdencies we will use in this hw are: FFMPEG, OpenSMILE, Python, sklearn, pandas, pytorch, librosa

Download OpenSMILE 3.0 from [here](https://github.com/audeering/opensmile/releases/download/v3.0.0/opensmile-3.0-linux-x64.tar.gz) and extract under the `./tools/` directory:
```
$ tar -zxvf opensmile-3.0-linux-x64.tar.gz
```
Install FFMPEG by:
```
$ apt install ffmpeg
```
Install python dependencies by (ignore if you start from .yaml): 
```
$ pip install scikit-learn==0.22 pandas tqdm librosa
```
If using conda, install pytorch by (ignore if you start from .yaml):
```
$ conda install pytorch torchvision torchaudio -c pytorch
```
For the last two parts of this hw, using gpu version of pytorch will significantly accelerate the feature extraction procedure. Please refer to [here](https://pytorch.org/get-started/locally/) for more detailed settings.

## Data and Labels
Please download the data from [AWS S3](https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip). Then unzip it and put the videos under "$path_to_this_repo/videos", and labels under "$path_to_this_repo/labels". You can either directly download the data to this folder or in anywhere else then build a [soft link](https://linuxhint.com/create_symbolic_link_ubuntu/)

The `.zip` file should include the following:
1. `video/` folder with 8249 videos in MP4 format
2. `labels/` folder with two files:
    - `cls_map.csv`: csv with the mapping of the labels and its corresponding class ID (*Category*)
    - `train_val.csv`: csv with the Id of the video and its label
    - `test_for_students.csv`: submission template with the list of test samples

## Task 1: MFCC-Bag-Of-Features
For this task we will provide code and instructions to extract MFCC-Bag-of-Features. Basiclly you don't need to modify any codes, just follow the instructions we listed below.

Firstly, let's create the folders to save extracted features, audios and models:
```
$ mkdir wav/ mfcc/ bof/ mp3/ snf/
```

Then extract the audio from the videos:
```
$ for file in videos/*;do filename=$(basename $file .mp4); ffmpeg -y -i $file -ac 1 -f wav wav/${filename}.wav; done
```
Tip 1: Learn the meaning of each option from [ffmpeg](https://ffmpeg.org/ffmpeg.html) (e.g. -ac, -f), you will need to change them in the later tasks.
Tip 2: It's possible that you can't extract audio files from all videos here, but the missing files should <10. Don't forget to report which video files had trouble and investigate the reason of the failure in the handout.

Then run OpenSMILE to get MFCCs into CSV files. We will directly run the binaries of OpenSMILE (no need to install):
```
$ for file in wav/*;do filename=$(basename $file .wav); ./tools/opensmile-3.0-linux-x64/bin/SMILExtract -C config/MFCC12_0_D_A.conf -I ${file} -O mfcc/${filename}.mfcc.csv;done
```

The above two steps may take 1-2 hours, depends on your device settings.

Then we will use K-Means to get feature codebook from the MFCCs as taught in class. Since there are too many feature lines, we will randomly select a subset (20%) for K-Means clustering by:
```
$ python scripts/select_frames.py --input_path labels/train_val.csv --ratio 0.2 --output_path mfcc/selected.mfcc.csv --mfcc_dir mfcc/
```

Then we train it by (50 clusters, this would take about 7-15 minutes):
```
$ python train_kmeans.py -i mfcc/selected.mfcc.csv -k 50 -o weights/kmeans.50.model
```

Finally, after getting the codebook, we will get bag-of-words features (a.k.a. bof).
```
$ python scripts/get_bof.py ./weights/kmeans.50.model 50 videos.name.lst --mfcc_path mfcc/ --output_path bof/
```

## Task 2: SVM classifier & MLP classifier
After Task 1, you should already extracted bof features under "./bof". We will use the features to train classifier. You will use sklearn to implement and train the classifiers. Different from previous task, we leave some blanks in the code and you need to finish them before running it. A `NotImplementedError` will be raised to notify you that you have some blanks didn't finished. Please remember to comment/remove these `NotImplementedError` lines after finishing the lines.

### SVM classifier
You need to fill in the blank left in `train_svm_multiclass.py`. The provided code has already loaded features and labels from files. You will need to convert them to a sklearn supported type and initialize and train the SVM classifier. The initial parameters you might use are: cache_size=2000, decision_function_shape='ovr', kernel="rbf". You are free to design your own initial params.

Then you can train the model by:
```
$ python train_svm_multiclass.py bof/ 50 labels/train_val.csv weights/mfcc-50.svm.model
```

Similarly, you also need to fill in the blank left in `test_svm_multiclass.py`. It requires you to load and get scores with the trained model.

Then get your predictions on test set by:
```
$ python test_svm_multiclass.py weights/mfcc-50.svm.model bof/ 50 labels/test_for_students.csv mfcc-50.svm.csv
```
mfcc-50.svm.csv is one of the files you need to upload in your final submission.

### MLP classifier
Similar to what you did in the previous section, you need to fill in the blank left in `train_mlp.py`. The initial parameters you might use are: chidden_layer_sizes=(512),activation="relu",solcer="adam",alpha=1e-3. You are free to design your own initial params.

Then train the model through:
```
$ python train_mlp.py bof/ 50 labels/train_val.csv weights/mfcc-50.mlp.model
```

Fill in the blank left in `test_mlp.py` and get your predictions on test set by:
```
$ python test_mlp.py weights/mfcc-50.mlp.model bof/ 50 labels/test_for_students.csv mfcc-50.mlp.csv
```
mfcc-50.mlp.csv is one of the files you need to upload in your final submission.

## Task 3: Extract SoundNet-Global-Pool
Although bof is a classic and successful method, the combination of big data and deep model has conquered the majority of classification tasks. In this task, we will use the [SoundNet](https://arxiv.org/pdf/1610.09001.pdf) to extract a vector feature representation for each video. You will see the giant improvment brought by this new feature. Meanwhile, we will use pytorch for this task. Since it will also be the major platform for next two assignments, it will be a great opportunity to make your hands dirty if you are not faimiliar with pytorch.

Go to `models/SoundNet.py`. `__init__(self):` is the initialization function of nn.Module class in pytorch. We usually design the structure of the module here. In this task, you need to complete the layer4-6 blocks according to the params provided in the [paper](https://arxiv.org/pdf/1610.09001.pdf). `forward():` is the function that tells the model how to process input data. In this task, you need to complete the processing steps of layer4-6. For the final part you need to finish `load_weights()` where it tells the model how to load params of pretrained weight.

When using deep learning backbones to extract features, you should be careful about the pre-processing steps of input data. Read the paper carefully, then re-extract audio files and store them under `./mp3`.

Then you can extract the features through:
```
$ python scripts/extract_soundnet_feats.py --feat_layer {the layer of feat you want to extract}
```
You should extract feature from which layer? Why?

The extracted features under `./snf` should have the same format as `./bof`. So you can reuse `train_mlp.py` and `test_mlp.py` to train and test an MLP classifier. Please remember to change the path and other args! Similar to previous task, store your results as `snf.mlp.csv` and this will be one of the files you need to submit.

## Task 4: Keep improving your model
The performance already get boosted a lot, however, it's far from an acceptable results. This is an open task, and you can try any methods you come about to improve your model. Some hints:

+ Split `trainval.csv` into `train.csv` and `val.csv` to validate your model variants. This is important since the leaderboard limits the number of times you can submit, which means you cannot test most of your experiments on the official test set.
+ Try different number of K-Means clusters
+ Try different layers of SoundNet
+ Try different classifiers (different SVM kernels, different MLP hidden sizes, etc.). Please refer to [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) documentation.
+ Try different fusion or model aggregation methods. For example, you can simply average two models' predictions (late fusion).


### Submission

You can then submit the test outputs to the leaderboard to evaulate your implementation:
```
https://www.kaggle.com/competitions/cmu-11775-f23-hw1-audio-based-med/overview
```
We use classification accuracy as the evaluation metric. Please refer to the `test_for_students.csv` file for submission format. You are expected to achieve 0.3+ for Task 2, 0.5+ for Task 3, and 0.7+ for Task 4.

