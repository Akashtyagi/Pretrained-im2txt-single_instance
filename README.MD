![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/releases/)
[![Build Status](https://travis-ci.org/anfederico/Clairvoyant.svg?branch=master)](https://travis-ci.org/anfederico/Clairvoyant)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)](https://bitbucket.org/lbesson/ansi-colors)

# Show and Tell: A Neural Image Caption Generator


This code can be integrated with an existing project and captioning model can be
load using object creation with loading the checkpoint just once.

* [x] Pretrained Model with 5 Million checkpoints.
* [x] Single instance loading.
* [x] Removed Bazel-bin dependenct.
* [x] Fixed vocab and checkpoints errors.



### Install Required Packages
First ensure that you have installed the following required packages:

* **Python 3.6** or greater
* **TensorFlow** Below 2.0(Ideal 1.15) ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](http://www.scipy.org/install.html))
* **Natural Language Toolkit (NLTK)**:
    * First install NLTK ([instructions](http://www.nltk.org/install.html))
    * Then install the NLTK data package "punkt" ([instructions](http://www.nltk.org/data.html))

### Steps to run

Step 1: ``` cd im2txt ```

Step 2: ``` python im2txt.py ``` 

### Results
![a person riding cycle](https://github.com/Akashtyagi08/Pretrained-im2txt-single_instance/blob/master/images/cycling.jpg)

```
Captions for image cycling.jpg:
  0) a person riding a bike on a city street (p=0.003317)
  1) a man riding a bike down a street . (p=0.000723)
  2) a person riding a bike on a street (p=0.000362)
```

Reference tensorflow model - [Tensorflow: Show and Tell](https://github.com/tensorflow/models/tree/master/research/im2txt)

If you find the repository useful, you can Star it as an appreciation.
