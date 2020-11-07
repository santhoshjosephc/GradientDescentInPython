# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:36:42 2020

@author: santhosh
"""


import pandas as pd
import pylab
import numpy as np
#df = pd.read_csv('C:\Users\santhosh\SanthoshPythonFile\Restaurant.csv', names=['x','y'])
df = pd.read_csv('Restaurant.csv',  names=['x','y'])

# Plot the variable to see the relationship. 

pylab.plot(df['x'], df['y'],'o')
x = df['x']
y = df['y']

#Alpha is the size of the baby step that we will take for each of the variables (the predictors). In this particular example, we only have one variable.
#You should run the program for various value of alpha and see the impact of it.  

alpha = 0.001
pylab.show())

#This function calculuates how much error is there in the prediction. Basically it just finds the square of the difference. If you dont square it, 
#a positive error and a negative error will cancel out each other which is not great. It give a false sense that the error is lower than what it actually is. 

def compute_cost_function(m, t0, t1, x, y):
    return 1/2/m * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])


#m is the number of records. 
    
m = len(df)

#t0 is the bias term. Eg: The linear equation for a line is y = px+ c where p is the slope and c is the intercept. The t0 here is same as c. 
# t1 is same as p

t0 = 0
t1=1

#costold is a variable that we will use to save the cost (error) in the last iteration so that we can compare with the current iteration. 
#This will help us understand if the cost is decreaing or incresing. 

costold = 0

#Initialize the cost for this iteration. Once we calculate it, we will compare it with the cost in the last iteration. 

cost =0
# grad0 is gradient for t0. ie: the amount by which we will change the t0 variable  (the intercept term). 
# grad1 is gradient for t1. ie: the amount by which we will change the x variable. 
grad0 = 0
grad1 = 0
# we are going to try this out 10000 times. We will keep checking how much the value is minimizing, if we are happy with the reduction in error,
# we will break from the loop.  
for j in range(10000):
    # this 'for loop' essentially finds the average value by which we will change the gradients.Why do we mulitiply the first gradient by 1 
    #and second one by X[i]? Answer: We are finding the partial derivative in order to minimize the function. Please read up on differntial calculus
    # and partial derivative if you dont understand what this step does and why it does that. It is too much to explain here. 
    for i in range(m):
        #grad0 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i]) ])
        #grad1 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) ])
        #grad0 = 1.0/m * sum([(t0 + t1*np.asarray(x[i]) - y[i]) ])
        #grad1 = 1.0/m * sum([(t0 + t1*np.asarray(x[i]) - y[i])*np.asarray([x[i]]) ])
        grad0 = grad0 + ((t0 + t1 * x[i]) - y[i])
        grad1 = grad1 + ((t0 + t1 * x[i]) - y[i]) * x[i]
        #grad1 = grad1 + ((t0 + t1 * x[i]) - y[i]) * 200
    grad0 = grad0/m
    grad1 = grad1/m

    #Now that we found the gradients, let us change the values of t0 and t1 and see how it impacts our cost (error)
    t0 = t0 - alpha * grad0
    t1 = t1 - alpha * grad1
    cost = compute_cost_function(10, t0, t1, x, y)
    print("cost: ", np.round(cost,2), " t0: ", round(t0, 2), " t1: ", round(t1, 2) )
    
    #If the change in cost (error), is less than .001, it has probably reduced the error as much as it can. Let us break from the loop.
    #What value you want to use here (instead of .001) is entirely upto you. 
    if abs(costold - cost) < .001:
        #printing the value of number of iterations for academic interest. 
        print ("number of iteration is: ", str(j))
        break;
    #saving the current cost to costold to be used in next iteration. 
    costold = cost
    
    #print("cost:", str(cost), " t0:", str(t0), " t1:",str(t1) )
    #print("cost:", str(cost))

