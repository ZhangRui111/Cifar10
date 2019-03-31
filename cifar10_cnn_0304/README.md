# cifar10_cnn_0304

### baseline
https://github.com/yscbm/tensorflow

### data set
CIFAR-10 binary version (suitable for C programs)

### data set path
./cifar10_0304/data/cifar-10-batches-bin/

### cnn architecture
Inputs -> 

conv1 -> pool1 -> norm1 ->

conv2 -> norm2 -> pool2 ->

full_layer1 ->

full_layer2 ->

dropout_layer ->

output_layer

### training result
step | train accuracy | test accuracy

i = 10000 | accuracy = 58%

i = 20000 | accuracy = 64%

...

i = 50000+ | accuracy = 97%~100% | 64%
