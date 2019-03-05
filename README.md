# VUT-Project
Image processing using Deep Learning

## Content

1. <a href="### Present work."> Relatórios </a>
2. <a href="### Tensorflow Object Detection API (TFODAPI)"> TFODAPI </a>

## Technical Report (tutorial)

### Present work.

1. Python code to load and [create the dataset](link to Github)
2. Classifier model which detects all 4 diseases.
3. Python code to automate the labeling process of the images prior to Object Detection. Almost 100% automated, except for the label


### How it was made and why.

The whole project is made on Deep Learning using Tensorflow (1.12.0) because it is a great way to deal with image processing without the need to understand deeply how image processing actually works. 

The main problems that were encountered were on preparing the dataset for training, because usually each deep learning API requires a different format of input, such as TFRecord file, on Tensorflow Object Detection API (add reference). One other common issue encountered was on fine tuning the model to get better accuracy.

So using Deep Learning really was a good to approach the problem without much previous knowledge.

### Tensorflow Object Detection API (TFODAPI)

I learned to prepare the dataset, creating the TFRecord file.
There are other ways to make it. But, the way I know:

1. Create a .yaml extension file, which is simply a plain text with .yaml at the end.
2. On this file, there should contain all information about the image bouding box:
    * Label
    * (x_min, y_min, x_max, y_max)
    * absolute path for the image (/home/user/example/example.jpeg)

This file will be used as input for the script that creates the TFRecordfile.
An example file can be found on my github, in the repository VUT Project.

### Where are the codes and information?
On my [github](https://github.com/Caduunb), inside the repository VUT Project.

### What is missing.

1. Understand why the classification model is stuck on the same accuracy it starts the training with.
2. Object Detection model.

### References
1. Series of videos from Andrew Ng on [Youtube](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
1. Series of videos from Geofrey Hinton, et al, on [Youtube](https://www.youtube.com/watch?v=cbeTc-Urqak&list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9\)
2. [Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015](http://neuralnetworksanddeeplearning.com/). 
3. Coursera [*Neural Networks and deep learning*](https://www.coursera.org/learn/neural-networks-deep-learning)
4. Coursera [*Machine Learning*](https://www.coursera.org/learn/machine-learning/home/welcome)6. Goodfellow's [book](http://www.deeplearningbook.org/)
7. Yoshua Bengo's [book](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf)
8. Tensorflow's [online tutorials](https://www.tensorflow.org/tutorials/)
11. Tensorflow Object Detection API, Blog Medium (https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)
14. Google AI [blog about Tensorflow O. Detection API](https://ai.googleblog.com/2017/06/supercharge-your-computer-vision-models.html)
9. Some good journals: [Springer](https://link.springer.com/), [IEEExplore](https://ieeexplore.ieee.org/Xplore/home.jsp), [Elsevier](https://www.elsevier.com/)
13. Reference people to follow on ML/Deep Learning: Andrew Ng, Hinton
