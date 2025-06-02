## Homework 1

### 1
Homework 01 deals with skip connections (also known as residual connections) and their benefits in neural network architectures like ResNet. 

The first task was to implement a simple skip-layer. The architecture should consist of two layers, 
with each layer containing a 2D convolution-layer, followed by batch-normalization- and ReLU as an activation function. 
To carry the input forward to complete the skip connection, we add it towards the output of the second layer in the forward pass. 
In this task, I assumed the output dimensions from the convolution should match the input dimensions, otherwise we would have to use for example 
a 1x1 convolution projection to match the shape of the output of the second layer to the shape of our input. 

### 2
In the second task we should replace the CNN used in the CNN demonstration notebook with either our own architecture carried forward from our 
skip-layer from task 1, or adapt the existing torchvision ResNet model to the sat6 dataset. I choose to adapt ResNet18, which involved:
1. Loading the model without weights
2. Changing the input channels of the first layer to 4, so it fits with our input data (R,G,B,NIR)
3. Changing the last layers output to 6 classes

The model was then trained on a local GPU to speed up training, but should not take that long on a CPU either.  

### 3
In the third task we calculated the Cohen's Kappa score, which is for example also used to measure inter-annotator agreement in annotation tasks,
and at last generated ROC curves for each of the classes and displayed them. The calculations were done by first binarizing (in this case the 
same as one-hot-encoding) the true and predicted labels and then using the roc_curve function from sklearn's metrics. 
