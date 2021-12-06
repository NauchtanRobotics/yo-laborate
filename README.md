# `yo-wrangle` 
_"helping computer vision engineers on a schedule"_

An app to efficiently wrangle images and labels (YOLO bounding box format) with special support for maintaining sample subsets.

![Editing short listed "most mistaken annotations"](yo-wrangle-55.png)

## Context and Purpose

Maintaining a dataset as a collection of independent subsets may be appropriate where there is some form of stratified sampling. For example, by geographical location, source camera/image quality/resolution, illumination conditions, or some other contextual grouping. Subgroups can be added as you bootstrap your dataset.  If model performance decreases after addition of any subset, the subset can easily be removed for a time in order to expedite training of a high performance model for a retricted context. `yo-wrangle` supports basic and advanced data wrangling with the option of maintaining subsets that can be selectively collated and split intro train-val sets.

`yo-wrangle` is particularly useful when working with the output from YOLOv5, in a Debian (linux) environment.

The most compelling value of yo-wrangle is integration of `fiftyone` to shortlist likely errors in bootstrapped training data and `OpenLabeling` for editing. No need to check `N` images, just review and edit the "most mistaken" at the click of a button.

## Mission Statement / Capability Statement 

An app (with graphical interface) that is great to help you build a superior object detection training dataset. Yo-wrangle makes it easy to refine top-shelf training data by offering:

* “Dataset Improvement" actions trigger opening of OpenLabeling and a FiftyOne interface simultaneously, loading them both with the the same short list of images that a most likely to contain labeling errors. 
* See the ground truth vs predictions in a high powered data exploration app whilst editing your annotations in another dedicated labelling app. 
* No more churning through copy-paste-search-open-edit-save sequences repeatedly for each and every image. 

Also, `yo-wrangle` provides dataset wrangling and bootstrapping basics, featuring:
* Wrangling tools that allow you to conserve precious NVM SSD space by sampling original images based on a cleansed directory of cropped image detections. 
* Options to uniformly subsample, filter according to confidence, restrict to certain classes/subfolders and amalgamate/recode class ids. 
* “Improve Dataset” action uses FiftyOne to find likely errors in old training data - no need to check `N` images. Just check the worst 20 say.  
* “Grow Dataset” action uses FiftyOne to mine for “hard” and unique training images. 
* Fewer “high value” training images leads to a compact dataset that performs well and is quicker to train. 
* Bootstrap a COCO dataset for segmentation modelling by converting bounding boxes to segmentation polygons.

## Installation

1. Install python-poetry then reboot. See https://python-poetry.org/docs/#installation
2. Clone this repository:  `git clone git@github.com:NauchtanRobotics/yo-wrangle.git`
3. Change working directory to newly cloned repo: `cd yo-wrangle`
4. Create virtual environment:  `poetry install` See https://python-poetry.org/docs/basic-usage/#installing-dependencies

For finding possible annotation errors with the help of `fiftyone-brain`, install python `torch` and `torchvision`
via instructions at https://pytorch.org/get-started/locally/

Installation of torch libraries will depend on the needs of your system. Having an RTX3090, I found
it best not to install CUDA / cuDNN on my system, but rather install python packages which contain
the necessary drivers for RTX3090, e.g. activate the poetry virtual env then run command in bash
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Troubleshooting tips for Linux Platform

* You need to reboot once between installing poetry and running `poetry install`
* I installed a package by lambda labs to help with driver dependencies for the GPU, but this may not be necessary. 
* To check GPU drivers are installed `sudo lshw -C display` or `hwinfo --gfxcard --short`
* Check you have tools for troubleshooting GPU operation by running this command `watch nvidia-smi`

## Getting Started

It would be great to get this working on Windows, however, I ran into trouble with running evaluations in `fiftyone`. 
Wrangling scripts will work fine on Windows.

There is no graphical interface yet, so edit the hard coded paths in `yo_wrangle.external_tool.test_find_errors()` then open a python console and type:
```
from yo_wrangle.external_tool import test_find_errors
test_find_errors()
```


## Contributing

### In-Scope
* TODO: GUI interface
* TODO: pytest unit tests
* TODO: Poetry virtual environment and packaging
* TODO: Continuous Integration scripts.
* TODO: Config.ini is parsed by config parser to provide/serialise various defaults such as class list path etc.

### Out-of-Scope
* ? open to discussion
* Web interface would be lovely, but there are no full time developers on this project - a little out of reach unless someone good at React is confident in integrating makesense..
* The app is not intended to be all things to all computer vision engineers. Be guided by the Context and Purpose, and Mission Statement.
