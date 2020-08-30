# Object classification with zero-shot learning
Image Processing Class (SCC5830) - Final Project

**Student:** Damares Oliveira de Resende

**#USP:** 11022990

This project consists of creating a computational model based on deep learning techniques to classify objects in an image with no previous knowledge related to that object. The idea is to train a neural network with a set **S** of classes and test it on a set **Z** of classes, where **Z** and **S** are disjoint.

The learning is based on the semantic features of each object and the dataset used is [Animals with Attributes 2](https://cvml.ist.ac.at/AwA2/). The algorithm receives two inputs: 1) an image; 2) an array with the semantic features; and gives one output: the class of the image. It is worth mentioning that a few images in the dataset can have more than one object.

The main tasks for this application are object detection, object segmentation, and feature extraction. To accomplish this, two computational models are used: 1) ResNet101 to extract image features; 2) regression model built from a deep neural network to correlate semantic features and visual features. The code is implemented based on Keras, a neural-network library written in Python and the areas of application is manyfold. This model can be used in any field where there is a lack of labeled images. 

An example of input image is given bellow:

![alt text](https://github.com/damaresende/USP/blob/master/SCC5830/prj/images/fox_10154.jpg)

More examples can be found [here](https://github.com/damaresende/USP/blob/master/SCC5830/prj/images/). The list of attributes can be found [here](https://github.com/damaresende/USP/blob/master/SCC5830/prj/images/AwA2-predicates.txt).

## Dataset Details

The dataset Animals with Attributes 2 consists of 37,322 images labeled in 50 different groups. In addition, each image is described semantically in 85 attributes. These attributes include descriptions of color, texture, shape, parts, and behavior of each animal class.

In order to simplify the application, 12 classes were chosen. Moreover, the dataset was reduced to 300 images per class, except for the class *spider+monkey*, which has only 291 images. Despite this minor difference, the dataset can be considered as a balanced dataset. In total, 3591 images are used. The classes chosen are the following:

Training and Validation sets (**S**):

<ul>
<li>dalmatian</li>
<li>persian+cat</li>
<li>horse</li>
<li>tiger</li>
<li>leopard</li>
<li>gorilla</li>
<li>chimpanzee</li>
<li>zebra</li>
<li>giant+panda</li>
</ul>

Test set (**Z**):
<ul>
<li>cow</li>
<li>lion</li>
<li>spider+monkey</li>
</ul>

The idea behind these classes is to use attributes that are shared among the chosen animals to describe other animals. What is expected from the model is that it will be able to classify unknown objects based on its semantic description. For instance, the neural network should be able to identify a *cow* by learning its shape and parts from the class *horse* and *zebra* and texture from the classes *dalmatian* and *giant+panda*. Similarly, a *lion* could be defined based on the association of characteristics of *tigers*, *leopards* and *persian+cats*.

In order to simplify the regression, the set of attributes was also reduced. From the 85 attributes available, 24 were chosen. Only characteristics that can be extracted from visual representations are considered, such as features that define color, shape, parts, and texture. Attributes that indicate behavior, food, environment, agility and relative size (big, small, thin) were disregarded. The list of attributes picked can be found below.

<ul>
<li>black</li>
<li>white</li>
<li>blue</li>
<li>brown</li>
<li>gray</li>
<li>orange</li>
<li>red</li>
<li>yellow</li>
<li>patches</li>
<li>spots</li>
<li>stripes</li>
<li>furry</li>
<li>hairless</li>
<li>toughskin</li>
<li>bulbous</li>
<li>bipedal</li>
<li>quadrupedal</li>
<li>longleg</li>
<li>longneck</li>
<li>flippers</li>
<li>hands</li>
<li>paws</li>
<li>tail</li>
<li>horns</li>
</ul>

## Feature Extraction

As mentioned before, in order to extract the visual features, the neural network **ResNet101** is used. The model takes as input the result of the layer immediately before the classification layer, which is a 1D array of length 2048. This network consists of a robust and deep architecture trained over the *ImageNet* dataset and has a high performance for classifying animals. 

The array of semantic characteristics is already provided by the authors of *Animals with Attributes 2* dataset, and can be found [here](https://cvml.ist.ac.at/AwA2/). It is important to mention that one set of characteristics is defined per class and not per image, or in other words, every image belonging to a class X has the same semantic descriptor. In this array, values ranging from **0** to **100** indicate that the animal has that probability of having that specific attribute, while number **-1** indicates that it does not have that feature. This probability is taken based on the number of people that described a particular animal with a specific set of characteristics.

## Computational Model

A schema of the deep learning model is given below, where the transfer learning technique is used. ResNet101 performs the heavy computation responsible for feature extraction, which consists of image processing operations such as morphology, segmentation, color and texture analysis, filtering, edge-detection, keypoint detection and etc. And the projection model is responsible for the clusterization of similar visual features and projection of them into a semantic space.

![alt text](https://github.com/damaresende/USP/blob/master/SCC5830/prj/images/zero_shot_model.png)

The idea is to train a regression model, here called projection model, to construct a function *f(x)* capable of predicting the semantic values *v* that represent the input image *x*. For that, a simple deep neural network was built using the Keras framework. The input is the set of visual features. The target values are the set of semantic attributes. This network is trained for 150 epochs with a batch size of 256, where 80% of the data is used for training and the remaining for validating the model. The summary of this model can be found [here](https://github.com/damaresende/USP/blob/master/SCC5830/prj/results/summary.txt).

Once *f(x)* is defined, zero-shot learning can be applied. In this context, a given image *I*, which class does not belong to the set of classes used to train the regressor, is used as input to the model. Then, the network predicts the image's set of semantic attributes *G*. Later, a simple *1-NN* classification model compares G to the set of semantic descriptors available by taking the Euclidean Distance between them. The one closest to *G* is chosen to represent that class, finally defining a label.

## Results and Conclusion

The figure below shows the results for classification accuracy and loss while training the model. These results indicate that the model converges. Training loss ends at around 22 and validation loss circa 453. Even though the values are high, for a regression model with 24 values to predict its performance is acceptable. In contrast, the training and validation accuracies reach 76%, which is a reasonable result.

![alt text](https://github.com/damaresende/USP/blob/master/SCC5830/prj/results/performance.png)

Despite the fact that the regression model converges during training when testing it against the unseen data, its performance drops to 1/10 of random classification (3.7037 %), which is an awful result. This indicates that the model is able to predict the classes that are known but is not generalized enough to classify unknown data. To mitigate that several approaches were tested, such as adding dropout layers to the neural network, increasing and decreasing its depth or number of neurons per layer, changing the loss and the optimizer used and changing from binary to continuous semantic representations. However, none of the approaches worked as expected.

One of the reasons for that to happen can be the oversimplification of the problem and the fact that the loss function is too restrictive. In future works, custom loss functions and different groups of classes and attributes will be tested seeking to improve this performance. In addition, other classifies other than *1-KK* can be used.

On the other hand, one interesting result was reached. For most cases, *cow* was misclassified with *horse*, *lion* was misclassified with *tiger*, *leopard* and *chimpanzee*, and *spider+monkey* was misclassified with *chimpanzee*, as can be observed from the [prediction results](https://github.com/damaresende/USP/blob/master/SCC5830/prj/results/prediction.txt). This result indicates that, except for the *lion* with *chimpanzee*, most classifications converges to an animal that is somehow similar to the one that the model is supposed to classify. Therefore, even though the zero-shot accuracy is not what expected, the model can indeed predict semantic characteristics for each animal.

## Code Details

The code is written in Python 3 and the framework Keras is used to apply deep learning algorithms. Source code can be found [here](https://github.com/damaresende/SCC5830/tree/master/src), and test code [here](https://github.com/damaresende/USP/blob/master/SCC5830/prj/test). Dataset, classes, labels and attributes definition can be found [here](https://github.com/damaresende/USP/blob/master/SCC5830/prj/data). Results are stored [here](https://github.com/damaresende/SCC5830/tree/master/results).

#### annotationsparser.py

This class is responsible for retrieving information related to the image semantic data, such as labels and attributes. It reads a matrix of predicates with the semantic description of each class and stores it in a pandas dataframe of shape (12, 85). The values are the floating numbers ranging from -1 to 100, and the indexes of the dataframe are the images original label, e.g. *20* for *gorilla* and *38* for *zebra*.

#### featuresparser.py

This class is responsible for reading the features. Function **get_visual_features** saves visual features in a numpy array of shape (3591, 2048), function **get_semantic_features** is a wrapper function for function **get_attributes** in  *annotationsparser.py*, which saves the the semantic dataframe in a numpy array of shape (3591, 24). Finally, function get_labels saves all labels IDs in a numpy of shape (3591,).

#### zeroshotmodel.py

Application main module. It loads the training and test sets, builds the regression model, trains it, encodes the testing visual features, applies the *1-NN* classifier to define the labels and save the results.

#### demo.py

Loads the model trained by *zeroshotmodel* script and tests it against the test set.

## References

Xian, Y., Lampert, C.H., Schiele, B. and Akata, Z., 2018. Zero-shot learning-a comprehensive evaluation of the good, the bad and the ugly. IEEE transactions on pattern analysis and machine intelligence.

Lampert, C.H., Nickisch, H. and Harmeling, S., 2009, June. Learning to detect unseen object classes by between-class attribute transfer. In 2009 IEEE Conference on Computer Vision and Pattern Recognition (pp. 951-958). IEEE.

Fu, Z., Xiang, T., Kodirov, E. and Gong, S., 2015. Zero-shot object recognition by semantic manifold distance. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2635-2644).

Xian, Y., Akata, Z., Sharma, G., Nguyen, Q., Hein, M. and Schiele, B., 2016. Latent embeddings for zero-shot classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 69-77).

Farhadi, A., Endres, I., Hoiem, D. and Forsyth, D., 2009, June. Describing objects by their attributes. In 2009 IEEE Conference on Computer Vision and Pattern Recognition (pp. 1778-1785). IEEE.
