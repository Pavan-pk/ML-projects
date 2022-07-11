# ML-projects
Projects from topics of Statistical machine learning (CSE 575)

# Classification using Naive Bayes and logistic regression
Dataset: The Dataset curated for this task consists of images of digits ‘7’ and ‘8’ from the MNIST set. Dataset is further divided into training and test data.<br>
1. ‘7’ consists of 6265 training and 1028 testing images, and they are labeled as 0.<br>
2. ‘8’ consists of 5851 training and 974 testing images, and they are labeled as 1.<br>
3. These images are not used as the input features for our classifier, we are using 2 features i.e a. The average of all pixel values in the image and b. The standard deviation of all pixel values in the image.<br>

### Naive Bayes Classification
We use the Bayes theorem to compute this probability.<br>
$$P(y/x) = αP(y)P(x/y)$$
Where,<br>
α - is a normalizing factor and can be ignored.<br>
P(y) - Prior distribution of the label in the training dataset, calculated by (No. of samples with label y)/(No. Total samples)<br>
P(x/y) - Posterior distribution given by, (No. of sample with ith feature taking value vk and label y)/P(Y)
#### Result:
1. Priors: $p(image=7)$ = 0.51, $p(image=8)$ = 0.48<br>
2. Prediction probability = $prior * prob of feature 1 (mean) * prob of feature 2 (standard deviation)$
3. Classification is done by comparison of each probabilities, i.e we predict the label to be 0(‘7’) if the probability of prediction 0(‘7’) is higher than probability of predicting 1(‘8’) and vice versa.
4. Prediction accuracy: '7' = 75.97%, '8' = 62.73%, total = 69.53%

### Logistic regression
In logistic regression:
a. Task is to predict $P(Y/X)$
$$P(Y/X) = \prod_{i=1}^n (\triangledown w^t x(i))^{y(i)} (1 - \triangledown w^t x(i))^{1-y(i)}$$
Where, <br>
𝑤 - is the weight matrix.<br>
∇𝑤 - delta of weight matrix changes.<br>
𝑥(𝑖) - ith sample of the training data.<br>
𝑦(𝑖) - label of ith sample.<br>

b. Use activation for output (sigmoid in our case) $/sigma {(z)} = \frac {1}{1 + e^{-z}}

c. learn the weights using gradient descent using the log likelihood of $$ ll = \sum_{i=1}^N y_i W^T x_i - log(1 + e^{W^Tx_i})$$
Where,<br>
W - weight matrix
𝑥𝑖 - ith sample of the training data.<br>
𝑦𝑖 - label of ith sample.<br>

Update weight using: $$ W^{k+1} = W^k + \eta {\triangledown}^{ll} $$
Where,
$W^{k+1}$ is the updated weight matrix (resultant after applying gradient)<br>
$W^k$ is the weight matrix used in this epoch.<br>
$\eta$ is the learning rate.<br>
${\triangledown}^{ll}$ is the delta update from log likelihood and is given by<br>
${\triangledown}^{ll} = X^T (Y-predictions)$

#### Results with Hyper parameters - epochs = 10,000, learning rate(η) = 1e-3<br>
1. Accuracy in prediction ‘7’ = 98.64%
2. Accuracy in prediction ‘8’ = 98.15%
3. Total accuracy = 98.40%

# K nearest neighbor
Concept of KNN with different initialization strategies and cluster size.<br>
Dataset: 2-D dataset with 300 samples.<br>
K: starts from 2 clusters to 10 clusters. <br>
Objective function: $$\sum_{i=1}^k \sum_{x \in Di} \Vert{x - \mu_i}\Vert$$
Strategy 1: Randomly pick k centroids from the dataset<br>
Strategy 2: Pick the first centroid randomly and k-1 centroids such that the datapoint on average is further away from all the centroids chosen so far.<br>
### Results
<img width="420" alt="Screenshot 2022-07-11 at 9 35 36 AM" src="https://user-images.githubusercontent.com/14234116/178313926-0ea37260-d315-4894-b484-cad19466dbae.png">
<img width="420" alt="Screenshot 2022-07-11 at 9 36 11 AM" src="https://user-images.githubusercontent.com/14234116/178313999-edfde0cf-a823-429b-adba-8f46c5f1dcf9.png">
<img width="420" alt="Screenshot 2022-07-11 at 9 36 45 AM" src="https://user-images.githubusercontent.com/14234116/178314071-1fb6fe28-61f1-4719-aafa-21ccabddf387.png">
<img width="420" alt="Screenshot 2022-07-11 at 9 37 10 AM" src="https://user-images.githubusercontent.com/14234116/178314136-4576d9e3-8b3a-4985-a248-fa552390f17b.png">

# CNN
Concepts of convolution neural networks.<br>
Dataset used: SVHN 32 X 32,<br>
The dataset consists of 32x32 RGB images, labels ranging from 0-9 which indicates the number in the image.<br>
Training dataset is of size 73,257 Testing database is of size 26,032<br>
### Model architecture:<br>
<img width="722" alt="model architecture" src="https://user-images.githubusercontent.com/14234116/178307885-cd7a1fae-e08f-4f7e-b44d-f77c673705bb.png"><br>
This is a simple CNN network for multiclass classification.<br>
The total number of parameters: 7022538.<br>
Each convolution layer has a kernel size of 5x5 and stride of 1 followed by a relu activation and max-pooling of 2x2 with 2 stride<br>
1st and 2nd conv have filter channels of 64 and 3rd one has 128 channels.<br>
There are 2 fully connected layers with 3072X2048 and 2048x10 input_output dimensions.<br>
Finally a softmax layer for 10 classes<br>


### Training hyper parameters:<br>
Learning rate: 0.01<br> Batch size: 64<br> Epochs: 20<br>

### Results

1. Training vs test loss for each epoch:<br>
<img width="359" alt="Screenshot 2022-07-11 at 9 10 01 AM" src="https://user-images.githubusercontent.com/14234116/178309130-c70dea8a-b4bd-4136-8f40-d9217168a41d.png"><br>

2. Accuracy 87.9<br>


