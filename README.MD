This code can be integrated with an existing project and captioning model can be
load using object creation with loading the checkpoint just once.


### Install Required Packages
First ensure that you have installed the following required packages:

* **Python 3.6** or greater
* **TensorFlow** 1.0 or greater ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](http://www.scipy.org/install.html))
* **Natural Language Toolkit (NLTK)**:
    * First install NLTK ([instructions](http://www.nltk.org/install.html))
    * Then install the NLTK data package "punkt" ([instructions](http://www.nltk.org/data.html))

### Steps to run

Step 1: ``` cd im2txt ```
Step 2: ``` python im2txt.py ``` 
```
Captions for image cycling.jpg:

##5 Million trained Iteration
Captions for image cycling.jpg:
  0) a person riding a bike on a city street (p=0.003317)
  1) a man riding a bike down a street . (p=0.000723)
  2) a person riding a bike on a street (p=0.000362)

##Captions for image phone.jpg:
  0) a close up of a cell phone on a table (p=0.001471)
  1) a cell phone sitting on top of a table . (p=0.001205)
  2) a close up of a cell phone on a wooden table (p=0.000246)


##2 Million trained Iteration
Captions for image cycling.jpg:
  0) a man riding a bike down a street next to a woman . <S> <S> . <S> . (p=0.000002)
  1) a man riding a bike down a street next to a woman . <S> <S> . <S> <S> (p=0.000002)
  2) a man riding a bike down a street next to a woman . <S> <S> . <S> <S> (p=0.000001)
```
