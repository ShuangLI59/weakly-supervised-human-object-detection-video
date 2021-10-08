## Weakly Supervised Human-Object Interaction Detection in Video via Contrastive Spatiotemporal Regions
This code is mainly based on our ICCV 2021 paper [Weakly Supervised Human-Object Interaction Detection in Video via Contrastive Spatiotemporal Regions](https://arxiv.org/abs/2110.03562).

Dataset webpage: https://shuangli-project.github.io/VHICO-Dataset/ <br>
Project webpage: https://shuangli-project.github.io/weakly-supervised-human-object-detection-video/ 


This project aims at weakly supervised human-object interaction detection in videos. 
We introduce a contrastive weakly supervised training loss that aims to jointly associate spatiotemporal regions in a video with an action and object vocabulary and encourage temporal continuity of the visual appearance of moving objects as a form of self-supervision. 

To train our model, we introduce a dataset comprising over 6.5k videos with human-object interaction annotations that have been semi-automatically curated from sentence captions associated with the videos.


### Install packages
```
conda install pytorch=0.4.1 cuda90 -c pytorch
pip install cython
pip install numpy scipy pyyaml packaging pycocotools tensorboardX tqdm scikit-image gensim
pip install opencv-python
pip uninstall matplotlib
conda install -c conda-forge matplotlib
pip uninstall pillow
conda install -c anaconda pil
```


## V-HICO Dataset
### Data
Because of licence issues, please download the corresponding videos from [Moments in Time Dataset](http://moments.csail.mit.edu/).
The data we used is from their extract frames with the folder name `video_256_30fps`.
For more information about our dataset, please visit the [dataset website](https://shuangli-project.github.io/VHICO-Dataset/).


### Data labelling tool
We used [LabelImg](https://github.com/tzutalin/labelImg) to annotate the human and object bounding boxes of video frames from the test set and the unseen test set.


## Evaluation
Please download the human annotations and saved results first. 
- [Human annotations](https://www.dropbox.com/s/owy9jvuj273j2n1/gt_annotations.zip?dl=0)
- [Saved results](https://www.dropbox.com/s/s8lf9zswydei7ud/cat%2BSpa%2BHum%2BTem%2BCon.zip?dl=0)

Please unzip the human annotations and put them in the `data` folder. <br>
Please unzip the saved results and put them in the `results` folder.


### Test set
	mAP: python eval/eval_vhico.py --eval_subset test --EVAL_MAP 1
	Recall: python eval/eval_vhico.py --eval_subset test --EVAL_MAP 0

### Unseen test set
	mAP: python eval/eval_vhico.py --eval_subset unseen --EVAL_MAP 1
	Recall: python eval/eval_vhico.py --eval_subset unseen --EVAL_MAP 0



## Training and Testing
### Prepare Data:
To train and test our model, please run the following codes:
- [Densepose](https://github.com/facebookresearch/DensePose) to extract human segmentation masks of the video frames.
- [word2vec](https://code.google.com/archive/p/word2vec/) to extract the features of action and object labels.

### Training:
	sh scripts/train_rel_mit.sh 

### Testing:
	sh test_rel_mit.sh








