"""
Start file for hw1pr2 of Big Data Summer 2017

Before attemping the problem, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

This file only has one part: the main driver.

First, fill in the solution you obtained from part (a) in part (c) of
the main driver to generate plot for part (c).

Please COMMENT OUT part(d) in main driver before you finish that step.
Otherwise, you won't be able to run the program because of errors.

Note:
1. You must finish the first two parts (math part) of this problem
   before attempting the coding part.

2. Please read the instructions and hints carefully, and use the name
   of the variables we provided, otherwise, the function may not work.

3. Placeholder values are given for the variables needed, they need to
   be replaced with your own code

3. Remember to comment out the TODO comment after you finish each
   part.
"""

import numpy as np
import matplotlib.pyplot as plt


###########################################
#           Main Driver Function          #
###########################################


if __name__ == '__main__':

    # part c: Plot data and the optimal linear fit
    # NOTE: to finish this part, you need to finish the part (a) of
    # this problem

    # load the four data points of tihs problem
    X = np.array([0, 2, 3, 4])
    y = np.array([1, 3, 6, 8])

    # plot four data points on the plot
    plt.style.use('ggplot')
    plt.plot(X, y, 'ro')

    m_opt = 62.0/35
    b_opt = 18.0/35

    theta = np.array([m_opt, b_opt]).transpose()

    num_pts = 100

    X_vals = np.linspace(-2,6,num=num_pts)
    X_space_stacked = np.array([X_vals, np.ones(X_vals.shape)])

    y_space = np.matmul(theta.transpose(), X_space_stacked)
    X_space, y_space = (
        X_space_stacked[0].reshape(-1,1),
        y_space.reshape(-1,1)
    )

    # plot the optimal learn fit obtained and save it to current dir
    plt.plot(X_space, y_space)
    plt.savefig('hw1pr2c.pdf', format='pdf')
    plt.close()




    # part d: Optimal linear fit with random data points

    # Parameters for Normal distribution
    mu, sigma, sampleSize = 0, 1, num_pts

    # normal distribution itself
    noise = np.random.normal(loc=mu, scale=sigma, size=sampleSize).reshape(-1,1)

    y_space_rand = y_space + noise

    # X_space_stacked = X_space # need to be replaced following hint 1
    # and 2
    a, b = (np.matmul(X_space_stacked, y_space_rand),
    np.matmul(X_space_stacked, X_space_stacked.transpose()))
    W_opt = np.linalg.solve(
        np.matmul(X_space_stacked, X_space_stacked.transpose()),
        np.matmul(X_space_stacked, y_space_rand)
    )

    # get the new m, and new b from W_opt obtained above
    b_rand_opt, m_rand_opt = W_opt.item(0), W_opt.item(1)

    theta_opt = np.array([b_rand_opt, m_rand_opt]).transpose()
    print(theta_opt)
    y_space_new = np.matmul(theta_opt.transpose(), X_space_stacked)
    y_pred_rand = (y_space_new.reshape(-1,1))

    # generate plot; plot original data points and line
    plt.plot(X, y, 'ro')
    orig_plot, = plt.plot(X_space, y_space, 'r')

    # plot the generated 100 points with white gaussian noise and the
    # new line
    plt.plot(X_space, y_space_rand, 'bo')
    rand_plot, = plt.plot(X_space, y_pred_rand, 'b')

    # set up legend and save the plot to the current folder
    plt.legend((orig_plot, rand_plot), \
               ('original fit', 'fit with noise'), loc = 'best')
    plt.savefig('hw1pr2d.pdf', format='pdf')
    plt.show()
    plt.close()
