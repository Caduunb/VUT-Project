# Detection and classification of retinal images
Image processing using Deep Learning on python3 Tensorflow API.

Author: Caio Eduardo Coelho de Oliveira, University of Brasília.

Supervisor: Martin Drahanský, Brno University of Technology.

---
### Present work.

1. Classification Model for 5 diabetic-retinopathy (DR) classes;
    - How to use Tensorboard to visualize a model.
2. Tensorflow Object Detection API (TFODAPI)
    - How to prepare a TFODAPI dataset out of raw images.


### How it was made and why?

The whole project is made on Deep Learning using Tensorflow (1.12.0) because it is a great way to deal with image processing without the need to understand deeply how image processing actually works. 

The main problems that were encountered were on preparing the dataset for training, because usually each deep learning API requires a different format of input, such as TFRecord file, on Tensorflow Object Detection API (add reference). One other common issue encountered was on fine tuning the model to get better accuracy.

So using Deep Learning really was a good to approach the problem without much previous knowledge on image processing.

For more details on the classification or object detection models, search on this repository's Wiki.

### Where is the code?

On my [github](https://github.com/Caduunb/VUT-Project), inside the repository VUT Project.

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

