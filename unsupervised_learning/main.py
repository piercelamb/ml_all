import sys


if __name__ == "__main__":
    passed_arg = sys.argv[1]
    if passed_arg.startswith('/'):
        dataroot = passed_arg
    else:
        dataroot = '/Users/plamb/Documents/Personal/Academic/Georgia Tech/Classes/ML/hw/unsupervised_learning/data/'
    if passed_arg == 'shoppers':
        run_shoppers(dataroot)
    elif passed_arg == 'ford':
        run_ford(dataroot)
    else:
        print("please run with an absolute path to the data")
        exit(146)