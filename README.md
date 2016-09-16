# TensorFlow-MNIST-example
Testing out TensorFlow examples for the MNIST dataset

## Installation
Create and activate a virtualenv for TensorFlow:

```shell
pyvenv env
source env/bin/activate
```

Set download URL for TensorFlow on Linux, Python3.5, CPU only and install it:

```shell
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
```

## Running

Execute `onelayer.py`.