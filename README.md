
# Capacity and tranability of Recurrent Neural Networks(rnn_avi)

# Motivation to problem statement
Number of bits an RNN , LSTM ,GRU can hold in their parameters.
# Illustrations
I have drawn a data set of binary inputs X and binary target labels Y. Inputs X has shape (nXb) and labels Y
has shape (1Xb), where b is number of samples which is treated as Hyperparameter(HP). Now I build different RNN
architectures using keras. Number of input neurons in my network is equal to dimensionality of input. As b is HP, by
varying b along with all HPs, to obtain the maximum mutual information(Bits), given by I (y;yˆ) = b + b (plogp +(1-p)
log(1-p)). Now i have performed similar tasks by varying number of parameters and finally plotted Bits vs parameters
plot and found that they have linear relationship. Also calculated bits-per-parameter, which on an average equals to 5
bits. 
# sources
Collins, Jasmine, Jascha Sohl-Dickstein, and David Sussillo. "Capacity and trainability in recurrent neural
networks." arXiv preprint arXiv:1611.09913 (2016).

# other information
Also gone through various research papers and found because of vanishing and exploding gradients, it make difficult for RNN to learn long term dependencies.
Solution to this problem is the use of rectified linear units, Identity initialization of weight
matrix, Hessian free optimization method , Long short-term memory(LSTM), more recently
Gated recurrent units(GRU).


