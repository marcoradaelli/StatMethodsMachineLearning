# StatMethodsMachineLearning
My work for Statistical Methods for Machine Learning exam assignment

In this file, a short description of the scripts is provided.

Dataset IS NOT included due to upload limits of GitHub. Original data from Kaggle (https://www.kaggle.com/moltean/fruits) should be put in folders Training and Test in the project folder.

prepare_dataset.py: the original classification is too specific. In order to implement our 10 categories, I use this script. It outputs in folders New_training and New_test.

learning_core.py: this script performs a simple learning procedure on the prepared datasets. Different NN architectures can be tried by uncommenting the corresponding lines.

convolutional_experiments.py: this script allows a simple analysis of the dependence of the "standard" CNN's performances on its hyper-parameters, as described. The script is designed to allow changing one hyperparameter at a time, and to re-train the NN with different choices for the hyper-parameters.

error_finder.py: this scripts have been used to perform the analysis about the kinds of errors made by the NN.

prepare_dislocated_dataset.py: this script prepairs the dataset by applying the random traslation used in the paragraph about pooling with CNNs.
displaced_test: training networks on dislocated dataset.
