import numpy as np


class DTLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, verbose=False, leaf_size=1):
        """
        Constructor method
        """
        self.leaf_size=leaf_size
        self.verbose=verbose
        self.decision_tree=None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "rlamb9"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        def build_tree(self, data_x, data_y):
            #given a np_array of data, find the index of the highest correlation with Y
            def best_feature_index(self, data_x, data_y):
                column_len = data_x.shape[1]
                max_correlation = 0
                max_index = 0
                for index in range(column_len):
                    correlation = np.correlate(data_x[:,index], data_y)
                    if correlation > max_correlation:
                        max_correlation = correlation
                        max_index = index
                return max_index

            #beginning of recursive tree build
            #check if the remaining data is less than or equal to the passed leaf_size so we can return
            final_row = data_x.shape[0]
            if final_row <= self.leaf_size:
                row = [-1, data_y.mean(), -1, -1]
                return np.array([row])
            #check if all the y data is the same
            elif np.all(data_y[0] == data_y[:]):
                row = [-1, data_y[0], -1, -1]
                return np.array([row])
            else:
                #get the best feature
                feature_index = best_feature_index(self, data_x, data_y)
                split_val = np.median(data_x[:, feature_index])

                #if we dont have values on either side of split_val, then we're at a leaf
                #this creates an array of bools and returns true if any True is found.
                if not np.any(split_val < data_x[:, feature_index]) or not np.any(split_val >= data_x[:, feature_index]):
                    row = [-1, data_y.mean(), -1, -1]
                    return np.array([row])

                #create the left tree
                left_remaining_values = data_x[:, feature_index] <= split_val
                left_tree = build_tree(self, data_x[left_remaining_values], data_y[left_remaining_values])
                #create the right tree
                right_remaining_values = data_x[:, feature_index] > split_val
                right_tree = build_tree(self, data_x[right_remaining_values], data_y[right_remaining_values])

                #add the root node
                root_row = [feature_index, split_val, 1, left_tree.shape[0] + 1]
                root = np.array([root_row])

                tree = np.concatenate((root, left_tree, right_tree), axis=0)
                return tree




        decision_tree = build_tree(self, data_x, data_y)
        self.decision_tree=decision_tree

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        estimate = []
        for row in points:
            node = 0
            leaf = -1
            while True:
                node = int(node)
                split_val = self.decision_tree[node][1]
                factor = self.decision_tree.item((node,0))
                if factor == leaf:
                    estimate = np.append(estimate, split_val)
                    break
                else:
                    if row[int(factor)] > split_val:
                        node = node + self.decision_tree.item((node,3))
                    else:
                        node = node + self.decision_tree.item((node,2))
        return estimate
