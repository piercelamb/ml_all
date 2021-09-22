# plamb's Machine Learning Assignments

### This repo contains four assignments covering:
### 1. [Supervised Learning](#supervised-learning) 
### 2. [Unsupervised Learning](#unsupervised-learning) 
### 3. Dimensionality Reduction,
### 4. Reinforcement Learning
***
## Supervised Learning
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

### - To run the code:
If you've cloned the repo, you'll need to download the datasets and put them in the
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

cd Documents/Personal/Academic/Georgia\ Tech/Classes/ML/hw/supervised_learning/

***
## Unsupervised Learning


