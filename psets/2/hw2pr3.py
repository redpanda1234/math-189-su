"""
Start file for hw2pr3 of Big Data Summer 2017

The file is seperated into two parts:
    1) the helper functions
    2) the main driver.

The helper functions are all functions necessary to finish the
problem. The main driver will use the helper functions you finished to
report and print out the results you need for the problem.

Before attemping the helper functions, please familiarize with pandas
and numpy libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

First, fill in the the code of step 0 in the main driver to load the
data, then please COMMENT OUT any steps in main driver before you
finish the corresponding functions for that step. Otherwise, you won't
be able to run the program because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. When filling out the functions below, remember to
    1) Let m be the number of samples
    2) Let n be the number of features

2. Please read the instructions and hints carefully, and use the name
   of the variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each
   part.
"""


#########################################################
#                        Helper Functions               #
#########################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#########################################
#                Part C                 #
########################################

def linreg(X, y, reg=0.0):
    """
    This function takes in three arguments:
        1) X, the data matrix with dimension m x (n + 1)
        2) y, the label of the data with dimension m x 1
        3) reg, the parameter for regularization

    This function calculates and returns the optimal weight matrix,
    W_opt.

    HINT: Find the numerical solution for part C
        1) use np.eye to create identity matrix
        2) use np.linalg.solve to solve for W_opt
    """
    # get identity matrix of the same shape as X.T @ X
    I = np.eye(X.shape[1])

    # remove the 0,0 entry to keep bias term unregularized
    I[0,0] = 0

    # Solve for optimal weight parameters minimizing ridge regression
    W_opt = np.linalg.solve(X.T @ X + reg*I, X.T @ y)
    return W_opt



def predict(W, X):
    """
    This function takes in two arguments:
        1) W, a weight matrix with bias
        2) X, the data with dimension m x (n + 1)

    This function calculates and returns the predicted label, y_pred.
    NOTE: You don't need to change this function.
    """
    return X * W



def find_RMSE(W, X, y):
    """
    This function takes in three arguments:
        1) W, a weight matrix with bias
        2) X, the data with dimension m x (n + 1)
        3) y, the label of the data with dimension m x 1

    This function calculates and returns the root mean-squared error,
    RMSE
    """
    # take the l2 norm
    RMSE = np.linalg.norm(y - X*W, ord=2)/np.sqrt(X.shape[0])
    return RMSE



def RMSE_vs_lambda(X_train, y_train, X_val, y_val):
    """
    This function takes in four arguments:
        1) X_train, the training data with dimension m x (n + 1)
        2) y_train, the label of training data with dimension m x 1
        3) X_val, the validation data with dimension m x (n + 1)
        4) y_val, the label of validation data with dimension m x 1

    This function generates a plot of RMSE vs lambda and returns the
    regularization parameter that minimizes RMSE, reg_opt.

    HINT: get a list of RMSE following the steps below:
        1) Constuct reg_list, a list of regularization parameters with
           random uniform sampling
        2) Generate W_list, a list of W_opt's according to regularization
           parameters generated above
        3) Generate, RMSE_list, a list of RMSE according to reg_list
    """
    reg_list = np.random.uniform(low=0.0, high=150.0, size=(300,))

    # Sort so that the plot can connect sequential terms nicely
    reg_list.sort()

    W_list = np.array([linreg(X_train, y_train, reg=x) for x in
        reg_list])
    # Calculate RMSE with the validation data
    RMSE_list = np.array([find_RMSE(W, X_val, y_val) for W in W_list])

    # Set up plot style
    plt.style.use('ggplot')

    # Plot RMSE vs lambda
    RMSE_vs_lambda_plot, = plt.plot(reg_list, RMSE_list)
    plt.setp(RMSE_vs_lambda_plot, color='red')
    plt.title('RMSE vs lambda')
    plt.xlabel('lambda')
    plt.ylabel('RMSE')
    plt.savefig('RMSE_vs_lambda.pdf', format='pdf')
    plt.close()
    print('==> Plotting completed.')

    reg_opt = reg_list[np.argmin(RMSE_list)]

    return reg_opt



def norm_vs_lambda(X_train, y_train, X_val, y_val):
    """
    This function takes in four arguments:
        1) X_train, the training data with dimension m x (n + 1)
        2) y_train, the label of training data with dimension m x 1
        3) X_val, the validation data with dimension m x (n + 1)
        4) y_val, the label of validation data with dimension m x 1

    This function generates a plot of norm of the weights vs lambda.

    HINT:
        1) You may reuse the code from RMSE_vs_lambda to generate
           w_list, the list of weights, and reg_list, the list of
           regularization parameters
        2) Then generate norm_list, a list of norm by calculating the
           norm of each weight
    """
    reg_list = np.random.uniform(low=0.0, high=150.0, size=(300,))

    # Sort so that the plot can connect sequential terms nicely
    reg_list.sort()

    W_list = np.array([linreg(X_train, y_train, reg=x) for x in
        reg_list])
    # Calculate RMSE with the validation data
    norm_list = np.array([np.linalg.norm(W, ord=2) for W in W_list])

    # Set up plot style
    plt.style.use('ggplot')

    # Plot norm vs lambda
    norm_vs_lambda_plot, = plt.plot(reg_list, norm_list)
    plt.setp(norm_vs_lambda_plot, color='blue')
    plt.title('norm vs lambda')
    plt.xlabel('lambda')
    plt.ylabel('norm')
    plt.savefig('norm_vs_lambda.pdf', format='pdf')
    plt.close()
    print('==> Plotting completed.')




#########################
#                Part D                 #
#########################

def linreg_no_bias(X, y, reg=0.0):
    """
    This function takes in three arguments:
        1) X, the data matrix with dimension m x (n + 1)
        2) y, the label of the data with dimension m x 1
        3) reg, the parameter for regularization

    This function calculates and returns the optimal weight matrix,
    W_opt and bias, b_opt seperately
    """
    t_start = time.time()

    # Get dim parameter for use everywhere
    m = X.shape[0]

    # get identity matrix of the same shape as X.T @ X
    I = np.eye(m)

    # get identity for gamma
    I2 = np.eye(X.shape[1])

    # Get 1 1.T
    ones = np.ones((m,m))

    # Calculate the middle term
    mat = I - (ones/m)

    lhs = (X.T @ mat @ X) + reg*I2

    rhs = X.T @ mat @ y

    # Solve for optimal weight parameters minimizing linear regression
    W_opt = np.linalg.solve(lhs, rhs)
    b_opt = np.ones((1,m)) @ (y - (X @ W_opt)) / m

    # Benchmark report
    t_end = time.time()
    print('--Time elapsed for training: {t:4.2f} seconds'.format(
            t=t_end - t_start ))

    return b_opt, W_opt




#########################
#                Part E                 #
#########################


def grad_descent(X_train, y_train, X_val, y_val, reg=0.0, lr_W=2.5e-12, \
                lr_b=0.2, max_iter=150, eps=1e-6, print_freq=25):
    """
    This function takes in ten arguments:
        1) X_train, the training data with dimension m x (n + 1)
        2) y_train, the label of training data with dimension m x 1
        3) X_val, the validation data with dimension m x (n + 1)
        4) y_val, the label of validation data with dimension m x 1
        5) reg, the parameter for regularization
        6) lr_W, the learning rate for weights
        7) lr_b, the learning rate for bias
        8) max_iter, the maximum number of iterations
        9) eps, the threshold of the norm for the gradients
        10) print_freq, the frequency of printing the report

    This function returns W, the optimal weight, and b, the bias by
    gradient descent.
    """
    m_train, n = X_train.shape
    m_val = X_val.shape[0]

    # Initialize weights array
    W = np.zeros((n,1))

    # Initialize weight gradient array
    W_grad = np.ones_like(W)

    # Initialize bias term
    b = 0.

    # ...and bias gradient. You might be sensing a pattern here.
    b_grad = 1.

    print('==> Running gradient descent...')

    # TODO: run gradient descent algorithm

    # HINT: Run the gradient descent algorithm followed steps below
    #     1) Calculate the training RMSE and validation RMSE at each
    #        iteration, and append these values to obj_train and obj_val
    #        respectively
    #     2) Calculate the gradient for W and b as W_grad and b_grad
    #     3) Upgrade W and b
    #     4) Keep iterating while the number of iterations is less
    #        than the maximum and the gradient is larger than the
    #        threshold

    obj_train = []
    obj_val = []
    iter_num = 0

    t_start = time.time()

    # start iteration for gradient descent
    while np.linalg.norm(W_grad) > eps and np.linalg.norm(b_grad) > eps \
    and iter_num < max_iter:
        train_RMSE = gd_RMSE(X_train, W, b, y_train)
        val_RMSE = gd_RMSE(X_val, W, b, y_val)

        obj_train += [train_RMSE]
        obj_val += [val_RMSE]

        W_grad, b_grad = get_gd_grads(X_train, W, b, y_train, reg=0.0)

        W -= lr_W * W_grad
        b -= lr_b * b_grad

        # print statements for debugging
        if (iter_num + 1) % print_freq == 0:
            print('-- Iteration{} - training RMSE {: 4.4f} - gradient norm {: 4.4E}'.format( iter_num + 1, train_RMSE, np.linalg.norm(W_grad)))

            # goes to next iteration
        iter_num += 1


    # Benchmark report
    t_end = time.time()
    print('--Time elapsed for training: {t:4.2f} seconds'.format(\
          t=t_end - t_start))


    # Set up plot style
    plt.style.use('ggplot')

    # generate convergence plot
    train_rmse_plot, = plt.plot(range(iter_num), obj_train)
    plt.setp(train_rmse_plot, color='red')
    val_rmse_plot, = plt.plot(range(iter_num), obj_val)
    plt.setp(val_rmse_plot, color='green')
    plt.legend((train_rmse_plot, val_rmse_plot), \
               ('Training RMSE', 'Validation RMSE'), loc='best')
    plt.title('RMSE vs iteration')
    plt.xlabel('iteration')
    plt.ylabel('RMSE')
    plt.savefig('convergence.pdf', format='pdf')
    plt.close()
    print('==> Plotting completed.')

    return b, W

def gd_RMSE(X, W, b, y):
    """
    Helper function for gradient descent algorithm. Takes X, a numpy
    array of data, W, a numpy array of weights, b, a bias term, and y,
    the expected value data thing. Returns the root mean square error
    of the model.
    """
    norm = np.linalg.norm((X @ W).reshape((-1, 1)) + b - y)
    RMSE = norm / np.sqrt(X.shape[0])
    return RMSE

def get_gd_grads(X, W, b, y, reg=0.0):
    """
    Helper function for gradient descent algorithm. Calculates the
    gradient of the weight array.
    """
    m, n = X.shape
    I = np.eye(n)
    lhsw = (X.T @ X + reg * I) @ W
    rhsw = X.T @ (b - y)
    W_grad = (lhsw + rhsw)/m

    b_grad = (sum(X @ W) - sum(y) + b*m)/m
    return (W_grad, b_grad)

###########################################
#               Main Driver Function              #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':

    # =============STEP 0: LOADING DATA=================
    print('==> Loading data...')

    # Read data LOCALLY! Remember to uncomment before uploading
    # though...

    df = pd.read_csv("online_news_popularity.csv", sep=", ",\
    engine="python")
    # df = pd.read_csv(\
    #     'https://math189su18.github.io/data/online_news_popularity.csv',\
    #     sep=', ', engine='python')

    # split the data frame by type: training, validation, and test
    train_pct = 2.0 / 3
    val_pct = 5.0 / 6

    df['type'] = ''
    df.loc[:int(train_pct * len(df)), 'type'] = 'train'
    df.loc[int(train_pct * len(df)) : int(val_pct * len(df)), 'type']\
    = 'val'
    df.loc[int(val_pct * len(df)):, 'type'] = 'test'


    # extracting columns into training, validation, and test data
    # Training data
    exclude = ['url', 'shares', 'type']

    X_train = np.array(df[df.type == 'train'][[col for col in \
                                     df.columns if col not in exclude]])
    y_train = np.log(df[df.type == 'train'].shares).values.reshape((-1, 1))

    # Validation data
    X_val = np.array(df[df.type == 'val'][[col for col in df.columns \
                                           if col not in exclude]])
    y_val = np.log(df[df.type == 'val'].shares).values.reshape((-1, 1))

    # Test data
    X_test = np.array(df[df.type == 'test'][[col for col in df.columns \
                                             if col not in exclude]])
    y_test = np.log(df[df.type == 'test'].shares).values.reshape((-1, 1))


    # TODO: Stack a column of ones to the feature data, X_train, X_val
    # and X_test

    # HINT:
    #     1) Use np.ones / np.ones_like to create a column of ones
    #     2) Use np.hstack to stack the column to the matrix
    "*** YOUR CODE HERE ***"
    X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
    X_val = np.hstack((np.ones((X_val.shape[0],1)), X_val))
    X_test = np.hstack((np.ones((X_test.shape[0],1)), X_test))
    "*** END YOUR CODE HERE ***"

    # Convert data to matrix
    X_train = np.matrix(X_train)
    y_train = np.matrix(y_train)
    X_val = np.matrix(X_val)
    y_val = np.matrix(y_val)
    X_test = np.matrix(X_test)
    y_test = np.matrix(y_test)



    # PART C
    # =============STEP 1: RMSE vs lambda=================
    # NOTE: Fill in code in linreg, findRMSE, and RMSE_vs_lambda for
    # this step

    print('==> Step 1: RMSE vs lambda...')

    # find the optimal regularization parameter
    reg_opt = RMSE_vs_lambda(X_train, y_train, X_val, y_val)
    print('==> The optimal regularization parameter is {reg: 4.4f}.'.format(reg=reg_opt))

    # Find the optimal weights and bias for future use in step 3
    W_with_b_1 = linreg(X_train, y_train, reg=reg_opt)
    b_opt_1 = W_with_b_1[0]
    W_opt_1 = W_with_b_1[1: ]

    # Report the RMSE with the found optimal weights on validation set
    val_RMSE = find_RMSE(W_with_b_1, X_val, y_val)
    print('==> The RMSE on the validation set with the optimal regularization parameter is {RMSE: 4.4f}.'.format(RMSE=val_RMSE))

    # Report the RMSE with the found optimal weights on test set
    test_RMSE = find_RMSE(W_with_b_1, X_test, y_test)
    print('==> The RMSE on the test set with the optimal regularization parameter is {RMSE: 4.4f}.'.format(RMSE=test_RMSE))



    # =============STEP 2: Norm vs lambda=================
    # NOTE: Fill in code in norm_vs_lambda for this step

    print('\n==> Step 2: Norm vs lambda...')
    norm_vs_lambda(X_train, y_train, X_val, y_val)



    # PART D
    # =============STEP 3: Linear regression without bias=================
    # NOTE: Fill in code in linreg_no_bias for this step

    # From here on, we will strip the columns of ones for all data
    X_train = X_train[:, 1:]
    X_val = X_val[:, 1:]
    X_test = X_test[:, 1:]

    # Compare the result with the one from step 1
    # The difference in norm should be a small scalar (i.e, 1e-10)
    print('\n==> Step 3: Linear regression without bias...')
    b_opt_2, W_opt_2 = linreg_no_bias(X_train, y_train, reg=reg_opt)

    # difference in bias
    diff_bias = np.linalg.norm(b_opt_2 - b_opt_1)
    print('==> Difference in bias is {diff: 4.4E}'.format(diff=diff_bias))

    # difference in weights
    diff_W = np.linalg.norm(W_opt_2 -W_opt_1)
    print('==> Difference in weights is {diff: 4.4E}'.format(diff=diff_W))



    # PART E
    # =============STEP 4: Gradient descent=================
    # NOTE: Fill in code in grad_descent for this step

    print('\n==> Step 4: Gradient descent')
    b_gd, W_gd = grad_descent(X_train, y_train, X_val, y_val, reg=reg_opt)

    # Compare the result from the one from step 1
    # Difference in bias
    diff_bias = np.linalg.norm(b_gd - b_opt_1)
    print('==> Difference in bias is {diff: 4.4E}'.format(diff=diff_bias))

    # difference in weights
    diff_W = np.linalg.norm(W_gd -W_opt_1)
    print('==> Difference in weights is {diff: 4.4E}'.format(diff=diff_W))
