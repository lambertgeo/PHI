DETECTION OF PHI STRUCTURES

This archive collects writings and programs dedicated to the detection and study of PHI structures.
PHIs are peculiar geological structures which have an higher-than-average potential to host mineral occurrences.

A draft paper, split in two parts introduces the concept of a PHI structure. It gives a brief history and lists methods to outline them (PHI patterns V3 part1 and part2).

OCTAVE APPLICATIONS:
The PHI structures can be detected on the basis of their geometrical components i.e. circles and lines segments. The “geometrical” detection methods rest on Hough transform for the circles and on Radon transform for the statistical treatment of the line segments.

Initial application programs were written in Octave, two are listed in "Octave" archive: 
-  hough.m
-  radon.m
Two application examples are given, they are self explanatory (additional info about their uses is in the draft paper)

The Octave programs are slow to implement and perform in a limited number of cases. They require prior knowledge of the river network.

DEEP LEARNING:

A large improvement was reached with the use of convolutional neural nets (CNNs) which return decent recognition stats.
Today the best results on PHIs identification reach 93% accuracy; the work is still going on to expand the image dataset and improve the accuracy.
The deep learning library chosen for the implementation of CNN's is Keras.  It is a collection of Python wrapper programs driving Theano or Tensorflow librarie.  The network used in this research has been built according to Chollet (2016) guidelines.

WITH THEANO BACKEND:
This was run on early versions of Keras version 1 and on Theano.
The adopted network uses the first four convolutional blocks of layers of Oxford's VGG16 model, pre-trained and frozen ; the fifth convolutional block, initially trained is re-used with the PHI images and a fifth dense block is used as a final classifier.
After preparing a original stock of 580 positive and 580 negative images, the data was augmented by flipping the images successively along horizontal and vertical axis. After a final visual review, the running stock was split into four datasets :
- a training dataset composed of 1820 positive and 1820 negative images and 
- a second validation dataset composed of  500 positives and 500 negatives  
( positive means the PHI structure is recognized over the image). 
The images, captured approximately square, were dimensioned at a  size of 224*224 pixels. The full set was run 200 times (epochs) for training with batches of 32. 
The buildup of the net goes in two steps:
- using the weights of lower 25 layers of VGG, unchanged;
- adding on the top two dense layers for the discrimination. (A 512 units layer was selected instead of a 256 units as proposed by Chollet for the dense_1 layer).
The accuracy of the identification process here reaches about 90% ; it is in the range of accuracies  related to automatic tissue classification from medical images on similar nets. 
Once trained, the net can be converted to a fully convolutional network by modifying the dense layers ( re-arranging their weights).
The net can then be used on larger images (e.g. 1200*600 pixels), which may cover a surface of 2 or 3 square degrees. The 224*224 pixels detector is scanned over the large image by the same convolutional process and produce a “heatmap” which indicates the best responses for location of the circular features. The scan proceeds along a method which mixes Blier (2016)and Perone(2016) programs .

Some layers respond particularly well to PHI characteristics; for instance layers 15 and 16 react to curved edges while layers 25 and 27 respond to circular “cloudy” features. Those four layers can be combined into a specific filter which can be applied over images covering several square degrees (hypercolumns: see Perone 2016). 
The final diagnostic layer (dense_2, 32)can be shown as well in the hypercolumns.
At last, the identified PHIs can be zoomed on, recorded with a plain screen capture tool and tested as 224*224 pixels images for a final diagnostic.
The stock of training/testing images and case examples of application are given.

WITH TENSORFLOW BACKEND
These tests are run on Keras version 2 and Tensorflow version 1.
The networks used are VGG16 and Xception.
VGG16 test was run on the same set of images as on the previous run i.e. on 224*224 images.

Xception is run on an augmented set of 299*299 images, a set of examples from European sites is given and tested: the accuracy of their diagnostic is about 93% ( see europa images/predictions europa.xls).



