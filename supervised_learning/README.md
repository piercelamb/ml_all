conda env create --file environment.yml
conda activate cs7641
either python main.py shoppers or
/Users/plamb/opt/miniconda3/envs/cs7641/bin/python main.py shoppers


Had to use columntransformer to get numerical values for my string columns
note that if using columntransformer, i cant inverse transform and get my strings back

I was worried about unbalanced data after splitting because 85% of sessions = FALSE for revenue
but printed the y_train result via print(y_train.value_counts()) which looked like:
False    7824
True     1423

After stratification:
False    7816
True     1431

I tried adding this to the column transformer: (RobustScaler(), numerical_columns)
it didnt change the accuracy at all

I tuned test_size as well and 0.3 gave the best result

BEFORE ADDING CROSS VALIDATION:
Accuracy Score on Overfitted DT: 0.8563087901394746
Size of overfitted DT: 1541
0.9010703859876743
Size of pruned DT: 15

AFTER ADDING CROSS VALIDATION:
0.8965348251913412 {'criterion': 'gini', 'max_depth': 5, 'splitter': 'best'}
Accuracy Score on cross-validated DT: 0.8965348251913412
Size of cross-validated DT: 61
Accuracy Score on pruned DT: 0.9080832657474993
Size of pruned DT: 13