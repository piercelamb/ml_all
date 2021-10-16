# plamb's Machine Learning Assignments

### This repo contains four assignments covering:
### 1. [Supervised Learning](#supervised-learning) 
### 2. [Unsupervised Learning](#unsupervised-learning) 
### 3. [Dimensionality Reduction](#dimensionality-reduction)
### 4. Reinforcement Learning

#### A note on commit messages: 
This is a single-committer homework repo with many pushes 
happening very early in the morning or late at night.
As such most of the messages are 'WIP' and do not reflect
how I operate in a team environment.
***
## Supervised Learning

### To run the code (for the impatient):

You should read all the context below, but if you've cloned the repo, you'll need to download the datasets and put them in the
`/supervised_learning/data` folder. The links above contain easy-to-find links to download.
Note Online Shopping Intention has a single CSV while Ford Alertness was already split into
Train/Test/Solution. All of these CSVs should be in `/data`

I installed [miniconda](https://docs.conda.io/en/latest/miniconda.html) in order to create
a unique environment for this runtime. I then created the below environment.yml file in /supervised_learning/:
```yml
name: cs7641
dependencies:
- python=3.9
- numpy=1.20.3
- matplotlib=3.4
- pandas=1.3.1
- scikit-learn=0.24.2
- pip 
- pip: 
  - pprofile==2.0.2 
  - jsons==0.8.8 
  - imbalanced-learn==0.8.0
```

Create the environment using this command (from /supervised_learning/)

`conda env create --file environment.yml`

Activate the environment with:

`conda activate cs7641`

At this point I would open `clean.py` in your editor to see which learner(s) is/are going to run.
To easily get there do a find for either `run_ford` or `run_shoppers`.
Once you're happy with what's running execute the below:

`path-to-conda-python clean.py shoppers-or-ford`

An example being:

`/Users/plamb/opt/miniconda3/envs/cs7641/bin/python clean.py ford`

## About the assignment:
For this assignment, we chose two datasets at random that contained 
classification problems. The goal was to optimally train five different supervised 
learners on the datasets and test how they performed on the classification 
problems. We then wrote a report to analyze how they performed, basing much
of the analysis on final scoring, wall clock time, validation curves and learning curves.
Ultimately we chose which learner performed the best.

The five learners we trained were:
1. Decision Tree
2. Boosted Decision Tree
3. Artificial Neural Network
4. K Nearest Neighbors
5. Support Vector Machine

When I chose my datasets in the first week of the class, I had zero context for
Machine Learning or how to choose a "good" dataset, so they were essentially chosen
at random. 

My first dataset was an Online Shopping Intention dataset [from UCI](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).
This dataset had a classification problem where various website interactions
were measured on an ecommerce site in order to classifiy whether or not a 
user would buy or not buy. The dataset contained approximately 17 features and 12,000 rows and
the classification problem was severely imbalanced (85% not buy).

My second dataset was the Ford Alertness Dataset from a [Kaggle competition](https://www.kaggle.com/c/stayalert/).
This dataset had a classification problem where physiological, environmental and vehicular
sensors were measured across ~100 drivers in order to classify if they were Alert or Not Alert.
The dataset contained approximately 32 features and 600k rows. The test set was pre-provided with 120k rows.
For this dataset, I found that it was only feasible (for some learners) on my laptop to run with 30% of the training set
and some preprocessing that reduced the number of features from 32 to 22. Since I had to do this
for one learner (SVM) I did it for all of them to make the comparison more consistent.


## About the code:
Note that this was my first time ever writing Machine Learning code. `main.py`
contains my very first (spaghetti) attempt where I learned a lot about sci-kit learn's
APIs. `clean.py` contains attempt #2 which is much cleaner, but still could be improved.
The idea behind `clean.py` was to be able to pass the absolute path of your data folder
to the file, which dataset you wanted to run on and it would call a function where
all five learners could easily be turned off or turned on via commenting and various
features of the learners could be tweaked or turned off/on also via commenting.

The code accepts two hyperparameters to optimize for each learner. The second parameter
will be initialized by `GridSearchCV`. Once initialized, the code will try every value 
of the other hyperparamter, training, cross-validating and tracking the score after each fold.
When complete it will select the best value. It will then perform the same tasks on the initialized
parameter. If the result of these tasks does not match the value produced by `GridSearchCV` it
will lock parameter 2 to the new result, and run all of this again. It will then produce 
validation curves for each parameter. The goal is to find the two optimal values for any dataset.

With the optimal hyperparameter values found, the code will now create a learner with those values,
fit the training set and generate predictions from the test set. All of this is then passed
to a `final_plots` function which will generate a confusion matrix, a loss curve (if ANN) and
a learning curve which compares the scoring method to the number of samples during training
and cross validation.

The code produces helpful print statements to let you know what it's doing.

Note that, for the shoppers dataset, you can set a `smote` tuple with associated floats that will
undersample the majority class and oversample the minority class according to your specification
(see `run_shoppers` function for more details)

Note that, for the ford dataset, you can pass different values to `get_data_ford` (inside `run_ford`)
that will reduce the number of features and reduce the size of the training set. 
See `get_data_ford` to get a sense of what these values can be and what they do.
Recall that I ran all of my learners on 30% of the training set with some preprocessing.

The code could be readily improved by not being rigidly bounded to two-parameter optimization
(e.g. it could loop over n parameters) and also by having a more readable-way of passing
around so many function parameters (perhaps using **args or **kwargs).

***
## Unsupervised Learning

### To run the code (for the impatient):

You should read all the context below, but if you've cloned the repo, you'll need to download the Ford Alertness dataset
`/unsupervised_learning/data` folder. The links above contain easy-to-find links to download.
Note the Ford Alertness data was already split into
Train/Test/Solution. All of these CSVs should be in `/data`

I installed [miniconda](https://docs.conda.io/en/latest/miniconda.html) in order to create
a unique environment for this runtime. I then created the below environment.yml file in /supervised_learning/:
```yml
name: randomized_optimization
dependencies:
- python=3.9
- numpy=1.20.3
- matplotlib=3.4
- pandas=1.3.1
- scikit-learn=0.24.2
- pip 
- pip: 
  - pprofile==2.0.2 
  - jsons==0.8.8
  - mlrose-hiive==2.2.3
```

Create the environment using this command (from /supervised_learning/)

`conda env create --file environment.yml`

Activate the environment with:

`conda activate randomized_optimization`

This repo contains `main.py` and `main2.py`. This occurred because I originally wrote all the code for this assignment
prior to the midterm, then after learning more about the assignment (e.g. averaging over random seeds) needed
to re-write part 1 to account for this learning. I decided to just create a new file `main2.py` to capture this. Because
of the need to re-write I was pressed for time and was not able to write as clean of code.

To run part 1:
`path-to-conda-python main2.py experiments`

To run part 2:
`path-to-conda-python main.py nn /your/dataroot`

(you can also change the dataroot in the code in the main function)
An example being:

`/Users/plamb/opt/miniconda3/envs/cs7641/bin/python main.py nn`

## About the Assignment:
For part 1 of this assignment we chose 3 fitness functions that would highlight differences among four randomized optimization
algorithms. The four RO algorithms we needed to highlight were Random Hill Climbing (RHC), Simulated Annealing (SA), 
Genetic Algorithms (GA) and MIMIC. The three fitness functions I chose were the OneMax Problem, The Four Peaks Problem
and The Flip Flop problem. Each problem is presented as a discrete optimization problem with binary number state vectors.

For part 2 of this assignment, we attempted to tune a Neural Network over one of our datasets from Supervised Learning
using RHC, SA and GA as the algorithm that will assign weights inside of it. Further, we train the NN with gradient descent
as well to use as a benchmark. I chose the Ford Alertness Dataset and reduced it in exactly the same way as I did for Supervised Learning.

mlrose_hiive's Runner classes are used in both parts to tune hyperparamters among a set of sensible parameter choices
and with an eye towards time complexity. Values are extracted from the Runner classes results for plotting and analysis.

## About the code:
As mentioned above, this assignment overlapped with a midterm making working on it a discontinuous endeavor.
My goal was to get the code all written and experiments run prior to studying for the midterm which I was able to do,
but as more office hours came out, more requirements were elucidated. As such, I had to re-write part 1 after the midterm which
is why `main2.py` exists and the code is a bit sloppy.

In essence, the part 1 code takes a list of fitness functions (onemax, fourpeaks, flipflop), a list of random seeds,
a list of RO algorithms and a list of problem sizes (problem sizes just are the size of the state vector in each case).
The code iterates the problem sizes, selecting one, then iterates the fitness funcs selecting one then iterates
the algorithms selecting one and finally iterates the random seeds selecting one. It runs the given alg's runner 
using the given fitness func and random seed and produces results (fitness, function evaluations and time). 
The results are combined across each seed and then averaged. These averaged results are plotted (fitness x iterations, fitness x function evals)
and then when all the algorithms finish, fitness x iterations is plotted for each on the same graph. Wall clock time is
written to a HTML file, capturing both time-to-peak fitness, and time to complete the experiment. Different plots are produced
for each problem size.

The part 2 code loads the Ford Alertness Dataset, reduces it as in Supervised Learning, takes a list of
algorithms and iterates them passing the dataset and algorithm to hiive's `NNGSRunner`. NNGSRunner uses
sklearn's `GridSearchCV` internally to tune hyperparameters. As such, a set of hyperparameters are passed
to it as well which include sensible values (learned from Supervised Learning) for default params (hidden layers, learning rate etc)
and sensible values for custom params (for e.g, RHC gets a 'restart_list' parameter). Here sensible is defined
as a tradeoff between improving the algorithms performance but also being wary of time complexity. Once 
NNGSRunner completes, a loss curve is generated over iterations and function evaluations. The best estimator is extracted from the underlying 
GridSearchCV object and used to generate a learning curve. Finally, wall clock times are written to a HTML file.

The code produces helpful print statements to let you know what is going on.

This code could definitely be improved, there are areas where looping could be used to make things more succint. 
Similarly to Supervised Learning, I could have split common functions into their own files/classes to make the code
cleaner and easier to grok. Again, this being my first foray into ML code, a lot of learning took place.

***
## Dimensionality Reduction

WIP