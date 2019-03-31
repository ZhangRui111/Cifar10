# Local version of official version. & 

# Add some explanation.

1. Running the simple CPU/GPU version.

__init__.py, cifar10.py, cifar10_input.py, cifar10_eval.py, cifar10_train.py are needed.

2. Be careful not to run the evaluation and training binary on the same GPU or else you might run out of memory. Consider running the evaluation on a separate GPU if available or suspending the training binary while running the evaluation on the same GPU.

> CIFAR-10 is a common benchmark in machine learning for image recognition.

> http://www.cs.toronto.edu/~kriz/cifar.html

> Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

> Detailed instructions on how to get started available at:

> http://tensorflow.org/tutorials/deep_cnn/

