# UNIGE-Machine-Learning 
Lab assignments developed during the Machine Learning course at UNIGE

1. [Linear classifier](/Assignment%201%20-%20linear-threshold%20classifier/Report/lab1_linearClassifiers.pdf)
   1. No algorithms developed, instead a manual-visual approach was used to select the hyperplane dividing 2 linearly-separable classes.
1. [Naive Bayes classifier](/Assignment%202%20-%20Naive%20Bayes%20classifier/Report/lab2_naiveBayesClassifier.pdf)
   1. Developed naive bayes classifier on categorical data (multivariate multinomial distribution).  
   1. Dataset used consisted of 14 observations with 4 features each.
   1. Binary classification (yes, no)
1. [k-Nearest Neighbors classifier](/Assignment%203%20-%20KNN%20classifier/Report/lab3_kNNClassifier.pdf)
   1. lazy-learner algorithm developed and tested on semeion-digits dataset (handwritten scanned digits). 
   1. Introduction to the confusion matrix and its terminology (accuracy, recall, precision, F1 score)
   1. t-SNE embedding (optional visualization usng built-in MATLAB function)
   1. Multiclass classification (0, 1, 2, ..., 9)
1. [Percetron and corss-validation](/Assignment%204%20-%20Perceptron%20CV/Report/lab4_perceptronClassifier.pdf)
   1. Developed Rosenblatt's perceptron using _sign_ as activation function.
   1. Tested with semeion-digits dataset (handwritten digits)
   1. Cross validation and hyper-parameter tuning.
   1. Binary (one-vs-all) and multiclass classification.
   1. Comparison between performance of perceptron and k-NN.
1. [Autoencoder using NNs](/Assignment%205%20-%20Perceptron%20CV/Report/lab5_autoencoder.pdf)
   1. Collaborative lab with Alexandre Sarazin
   1. Use of MATLAB's nntool (neural networks toolbox)
   1. Autoencoder using a simple Neural Netwrok with 1 hidden layer.
  
## Klassifiers
(cleverly intended misspelling)

This folder contains the implementation of
* kNN classification algorithm
* Rosenblatt's perceptron

## common-functions

Useful functins used throghout the reports
* Confusion matrix related
  * plot_confMat.m
  * accuracy.m
  * precision.m
  * recall.m
  * F1Score.m
  * specificity.m
* Cross-validation
  * cross_val_score.m
  * get_scores.m
  * merge_scores.m
* Train-test-split
  * train_test_split.m
  * stratified_split.m
