
# 1.Capacity and tranability of Recurrent Neural Networks(rnn_avi)

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
#########################################
# Techniques For improving Success rates of Recto - Viginal Artificial insemination using Hidden markov models:
# Motivation to problem statement:
We consider a system where there are multiple timescales at which Hidden Markov Models are valid. When the
difference between the different timescales is huge and when the number of parameters given to the Baum-Welch
learning algorithm is not provided correctly, there are significant chances that the learning algorithm may converge to
a wrong transition probability matrix.For example, in dairy cow monitoring, there are at least two HMMs, one over
the (feeding, not feeding) states and another over the (Heat, Not Heat) state. For this system the observations are the
accelerometer data obtained over time by, say, periodic sampling of the accelerometer.So we need to choose initial
model parameters such as number of hidden states (m) and number of underlying timescales (K) so that B-W algorithm
captures correct time scale dynamics.
## Elbow method:
## Akaike information criterion (AIC) and Bayesian Information Criterion (BIC)
## Learning by Baum-Welch (B-W) algorithm
Assume K the number of time scales is known.We can immediately observe that this mixture can itself be viewed
as a single "composite" HMM where the transition matrix A of the model is block-diagonal.where the initial state
probabilities are chosen appropriately and initially randomly choosing either the "upper" matrix A1 (with probability P)
or the "lower" matrix A2 with probability 1 -P and learning proceeds directly on the composite HMM (with matrix A)
in the usual Baum-Welch fashion using all of the sequences.
## sources:
1)Bicego, Manuele, Vittorio Murino, and Mário AT Figueiredo. "Similarity-based clustering of sequences using
hidden Markov models." International Workshop on Machine Learning and Data Mining in Pattern Recognition.
Springer, Berlin, Heidelberg, 2003..
2) Šabata, Tomáš, Tomáš Borovicka, and Martin Holena. "Modeling and Clustering the Behavior of Animals Using
Hidden Markov Models." CEUR Workshop Proceedings 1649. 2016. .
3) Smyth, Padhraic. "Clustering sequences with hidden Markov models." Advances in neural information processing
systems. 1997. 



