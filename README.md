
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
# 2)Techniques For improving Success rates of Recto - Viginal Artificial insemination using Hidden markov models:
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
![hmm_elbow](https://user-images.githubusercontent.com/65336197/103477055-df578500-4de0-11eb-9483-a195409e755e.JPG)

## Akaike information criterion (AIC) and Bayesian Information Criterion (BIC):
![hmm_bic](https://user-images.githubusercontent.com/65336197/103477058-e383a280-4de0-11eb-93b8-c4c0aaa25110.JPG)

## Learning by Baum-Welch (B-W) algorithm
Assume K the number of time scales is known.We can immediately observe that this mixture can itself be viewed
as a single "composite" HMM where the transition matrix A of the model is block-diagonal.where the initial state
probabilities are chosen appropriately and initially randomly choosing either the "upper" matrix A1 (with probability P)
or the "lower" matrix A2 with probability 1 -P and learning proceeds directly on the composite HMM (with matrix A)
in the usual Baum-Welch fashion using all of the sequences.
![hmm_b_w](https://user-images.githubusercontent.com/65336197/103477066-eaaab080-4de0-11eb-901a-34fcef2a30c9.JPG)
## Future work:
investigate the link between the rate at which events are observed by a monitoring system.
Convert observations to voltages of specified limit using IOT devices and monitor system performance.
## sources:
1)Bicego, Manuele, Vittorio Murino, and Mário AT Figueiredo. "Similarity-based clustering of sequences using
hidden Markov models." International Workshop on Machine Learning and Data Mining in Pattern Recognition.
Springer, Berlin, Heidelberg, 2003..
2) Šabata, Tomáš, Tomáš Borovicka, and Martin Holena. "Modeling and Clustering the Behavior of Animals Using
Hidden Markov Models." CEUR Workshop Proceedings 1649. 2016. .
3) Smyth, Padhraic. "Clustering sequences with hidden Markov models." Advances in neural information processing
systems. 1997. 

# 3.UNIVERSAL RAIL MILL (URM) Data Analysis:
# Motivation to problem:
The Bhilai steel plant(BSP) located in Bhilai ,in Indian state of Chhattisgarh,is India's first and main producer of
steel rails,as well as major producer of wide steel plates and other steel products.The plant is the sole supplier
of the country’s longest rail tracks,which measures 260 meters(850ft).The 130-meter rail,which would be the
world’s longest rail line in a single piece, was rolled at URM ,Bhilai Steel Plant on 29 November 2016.URM
section in BSP is operating in three shifts A,B and C .The efficiency of 130 meter rail production in URM section
is only around 60 % and BSP want to improve it to around (85-90)%.In this report I have mentioned how the
efficiency of 130-meter rail production is varying in these three shifts and how to analyse various parameters
which mainly effect in URM section of BSP to achieve 130-meter rail production.
# PROBLEM STATED BY URM TEAM OF BSP:
a.To improve 130 m rail production and to analyse various parameters to achieve this. 
b.Current output is 60 % (approx) and End goal is to achieve (85-90) %(approx).
# PROCESS IN URM:
1. CHARGING 2. BREAKDOWN (1) 3 .BREAKDOWN (2) 4. TANDEM MILL 5.VISUAL INSPECTION
We received two data sets: a. Tandem mill data b. Visual inspection data
# PRELIMINARY OBSERVATIONS FROM TANDEM MILL DATA:
a. Each file is dedicated to a rail. b. Each rail is identified with a time-stamp.
# PROBLEMS IDENTIFIED IN DATASETS PROVIDED:
a. ID MISMATCH between Tandem mill data and Visual inspection data i.e. same id Ex: A001 is given to first rail after 6 AM in both Tandem mill and Visual inspection data. b. We haven’t received any data regarding Cooling Bed, Breakdown(1), Breakdown(2) as they help in
correlating data between Tandem mill and Visual inspection.
![bsp_1](https://user-images.githubusercontent.com/65336197/103477283-6e659c80-4de3-11eb-9564-b3f08ca89c2d.JPG)
![bsp_2](https://user-images.githubusercontent.com/65336197/103477286-7291ba00-4de3-11eb-83f7-8fc82a887213.JPG)
# CONCLUSION FROM VISUAL INSPECTION DATA:
a. Increasing trend in efficiency from shift A, B and C. b. But at present we do not know the parameters which affect this increasing efficiency trend.
# BY USING IBA ANALYSER - PLOTTED GRAPHS ANALYSING PARAMETERS SUCH AS :
![bsp_3](https://user-images.githubusercontent.com/65336197/103477288-7cb3b880-4de3-11eb-93f6-348ba55dbc98.JPG)


