
# Capacity and tranability of Recurrent Neural Networks(rnn_avi)

# Motivation to problem statement
The motivation came from recently published article on capacity of synapses in biological
neurons which is 4.7 bits per synapses. Now i tried to calculate number of bits an RNN , LSTM ,GRU can hold in their parameters.They provide an estimate of the number of bits which a weight may be compressed without loss in task performance. 
# Illustrations
I have drawn a data set of binary inputs X and binary target labels Y. Inputs X has shape (nXb) and labels Y
has shape (1Xb), where b is number of samples which is treated as Hyperparameter(HP). We are assuming Rnn to be a similar to binary symmetric cahnnel. Now I build different RNN
architectures using keras. As b is HP, by varying b along with all HPs, to obtain the maximum mutual information(Bits), given by I (y;yˆ) = b + b (plogp +(1-p)
log(1-p)). Now i have performed similar tasks by varying number of parameters and finally plotted Bits vs parameters
plot and found that they have linear relationship. Also calculated bits-per-parameter, which on an average equals to 5
bits. 
# Result:
![rnn_result](https://user-images.githubusercontent.com/65336197/103476592-4b37ee80-4ddd-11eb-8d34-318cff62c4e4.JPG)


# other information
Also gone through various research papers and found because of vanishing and exploding gradients, it make difficult for RNN to learn long term dependencies.
Solution to this problem is the use of rectified linear units, Identity initialization of weight
matrix, Hessian free optimization method , Long short-term memory(LSTM), more recently
Gated recurrent units(GRU).
# sources
Collins, Jasmine, Jascha Sohl-Dickstein, and David Sussillo. "Capacity and trainability in recurrent neural
networks." arXiv preprint arXiv:1611.09913 (2016).
 Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint
arXiv:1412.6980 (2014).
Le, Quoc V., Navdeep Jaitly, and Geoffrey E. Hinton. "A simple way to initialize recurrent networks of rectified
linear units." arXiv preprint arXiv:1504.00941 (2015).

