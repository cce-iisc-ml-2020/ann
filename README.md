
# Multi Layer Neural Network
Neural networks are a class of machine learning algorithms used to model 
complex patterns in datasets using multiple hidden layers and activation 
functions.

# DataSet
(http://archive.ics.uci.edu/ml/datasets/banknote+authentication)
The Banknote Dataset involves predicting whether a given banknote is authentic 
given a number of measures taken from a photograph.

It is a binary (2-class) classification problem. There are 1,372 observations 
with 4 input variables and 1 output variable. The variable names are as follows:

Variance of Wavelet Transformed image (continuous).
Skewness of Wavelet Transformed image (continuous).
Kurtosis of Wavelet Transformed image (continuous).
Entropy of image (continuous).
Class (0 for authentic, 1 for inauthentic).

# Run: 
# python3 main-4i-4h-1o.py

--- output ---
# python3 main-4i-4h-1o.py

Stage 1) Random starting weights: 
# Layer 1 (4 neurons, each with 4 inputs):
[[4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01]
 [1.46755891e-01 9.23385948e-02 1.86260211e-01 3.45560727e-01]
 [3.96767474e-01 5.38816734e-01 4.19194514e-01 6.85219500e-01]
 [2.04452250e-01 8.78117436e-01 2.73875932e-02 6.70467510e-01]]
# Layer 2 (1 neuron, with 4 inputs):
[[0.4173048 ]
 [0.55868983]
 [0.14038694]
 [0.19810149]]

Stage 2) New weights after training: 
# Layer 1 (4 neurons, each with 4 inputs):
[[-0.80882082  0.44127317 -1.95958137  2.30771812]
 [ 0.20107878 -0.26749924  0.34160953 -0.19752144]
 [ 0.17624295  0.71150346 -0.08378137  0.32611257]
 [ 0.29950621  0.95467863  0.59970368 -0.03013834]]
# Layer 2 (1 neuron, with 4 inputs):
[[ 1.07345495]
 [-0.16503406]
 [ 3.97252318]
 [-4.81665941]]

Testing: Expectation O = [1]
Input: [-2.4941  3.5447 -1.3721 -2.8483]
Output: [0.99201085]

Testing: Expectation O = [0]
Input: [ 3.9362 10.1622 -3.8235 -4.0172]
Output: [0.00860221]

--- end output ---
