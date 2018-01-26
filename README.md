# UNIGE-Machine-Learning 
Lab assignments developed during the Machine Learning course at UNIGE

## Requirements
* MATLAB
* LaTeX

## Assignments summary

1. [Linear classifier](/Assignment%201%20-%20linear-threshold%20classifier/Report/lab1_linearClassifiers.pdf)
   * No algorithms developed, instead a manual-visual approach was used to select the hyperplane dividing 2 linearly-separable classes.
1. [Naive Bayes classifier](/Assignment%202%20-%20Naive%20Bayes%20classifier/Report/lab2_naiveBayesClassifier.pdf)
   * Developed naive bayes classifier on categorical data (multivariate multinomial distribution).  
   * Dataset used consisted of 14 observations with 4 features each.
   * Binary classification (yes, no)
1. [k-Nearest Neighbors classifier](/Assignment%203%20-%20KNN%20classifier/Report/lab3_kNNClassifier.pdf)
   * lazy-learner algorithm developed and tested on semeion-digits dataset (handwritten scanned digits). 
   * Introduction to the confusion matrix and its terminology (accuracy, recall, precision, F1 score)
   * t-SNE embedding (optional visualization usng built-in MATLAB function)
   * Multiclass classification (0, 1, 2, ..., 9)
1. [Percetron and corss-validation](/Assignment%204%20-%20Perceptron%20CV/Report/lab4_perceptronClassifier.pdf)
   * Developed Rosenblatt's perceptron using _sign_ as activation function.
   * Tested with semeion-digits dataset (handwritten digits)
   * Cross validation and hyper-parameter tuning.
   * Binary (one-vs-all) and multiclass classification.
   * Comparison between performance of perceptron and k-NN.
1. [Autoencoder using NNs](/Assignment%205%20-%20Neural%20Networks/Report/lab5_autoencoder.pdf)
   * Collaborative lab with Alexandre Sarazin
   * Use of MATLAB's nntool (neural networks toolbox)
   * Autoencoder using a simple Neural Netwrok with 1 hidden layer.
  
## Klassifiers
(cleverly intended misspelling)

This folder contains the implementation of
* kNN classification algorithm
* Rosenblatt's perceptron

## common-functions

Useful functions used throghout the reports
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
