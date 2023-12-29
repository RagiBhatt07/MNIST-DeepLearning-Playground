# MNIST-DeepLearning-Playground

# Practical classes


All exercices will be in Python. It is important that you keep track of exercices and structure you code correctly (e.g. create funcions that you can re-use later)

We will use Jupyter notebooks (formerly known as IPython). You can read the following courses for help:
* Python and numpy: http://cs231n.github.io/python-numpy-tutorial/
* Jupyter / IPython : http://cs231n.github.io/ipython-tutorial/


# Neural network: first experiments with a linear model

In this first lab exercise we will code a neural network using numpy, without a neural network library.
Next week, the lab exercise will be to extend this program with hidden layers and activation functions.

The task is digit recognition: the neural network has to predict which digit in $\{0...9\}$ is written in the input picture. We will use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, a standard benchmark in machine learning.

The model is a simple linear  classifier $o = \operatorname{softmax}(Wx + b)$ where:
* $x$ is an input image that is represented as a column vector, each value being the "color" of a pixel
* $W$ and $b$ are the parameters of the classifier
* $\operatorname{softmax}$ transforms the output weight (logits) into probabilities
* $o$ is column vector that contains the probability of each category

We will train this model via stochastic gradient descent by minimizing the negative log-likelihood of the data:
$$
    \hat{W}, \hat{b} = \operatorname{argmin}_{W, b} \sum_{x, y} - \log p(y | x)
$$
Although this is a linear model, it classifies raw data without any manual feature extraction step.
