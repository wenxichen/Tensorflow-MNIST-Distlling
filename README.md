# Tensorflow-MNIST-Distlling

Simple tensorflow implementation of distilling in MNIST.

### Installation ### 

* link the mnist folder in tensorflow: `ln -s tensorflow/tensorflow/examples/tutorials/mnist mnist`
* run large_net.py to create a large net
* run small)net.py to create a small net
* run distill.py to distill the knowledge from the large net to a trained small net or a newly intitialized small net