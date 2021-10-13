# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:12:25 2021

@author: const
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

def create_dataset(w_star, x_range, sample_size, sigma, seed= None):
    random_state = np.random.RandomState(seed)
    #array of random numbers (drawn from unifrom distr) of lingth sample size
    x = random_state.uniform(x_range[0], x_range[1], (sample_size))
    X = np.zeros((sample_size, w_star.shape[0]))
    for i in range(sample_size):
        X[i,0] = 1
        for j in range(1, w_star.shape[0]):
            X[i,j] = x[i]**j
    y = X.dot(w_star)
    if sigma > 0:
        y += random_state.normal(0.0, sigma, sample_size).reshape(-1,1)
    return X,y

        
def linRegression(X_tr,y_tr, X_val, y_val, alpha = 0.11, T = 800):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #coefficients to train
    w_star = np.zeros((X_tr.shape[1],1))
    #learning rate
    #alpha = 0.011
    #training steps
    #T = 800
    #feature dimension
    in_features = X_tr.shape[1]
    out_features = 1
    model = nn.Linear(in_features, out_features, bias=False)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = alpha)
    #Prepare inputs for nn - Linear
    X_tr = torch.from_numpy(X_tr).float().to(DEVICE)
    X_val = torch.from_numpy(X_val).float().to(DEVICE)
    y_tr = y_tr.reshape((-1,1))
    y_tr = torch.from_numpy(y_tr).float().to(DEVICE)
    y_val = y_val.reshape((-1,1))
    y_val = torch.from_numpy(y_val).float().to(DEVICE)
    
    #training
    for step in range(T):
        #training
        model.train()
        optimizer.zero_grad()
        y_ = model(X_tr)
        loss = loss_fn(y_, y_tr)
        #print(f"Step{step}: train loss{loss}")
        loss.backward()
        optimizer.step()
        #evaluation
        model.eval()
        with torch.no_grad():
            y_ = model(X_val)
            val_loss = loss_fn(y_, y_val)
        #print(f"Step {step}: val loss: {val_loss}")
    w_star = model.weight
    return [w_star, val_loss] 

def findHyperparameter(X_training, y_training, X_validation, y_validation):
    #after some simple test with I realize the learning rate should be around 0.01
    alphas = np.array([0.1, 0.02, 0.015, 0.012, 0.011, 0.01, 0.009, 0.008, 0.005])
    a_size = np.size(alphas)
    Ts = np.array([100, 300, 500, 800, 1000, 3000])
    T_size = np.size(Ts)
    val_Loss = np.zeros((a_size, T_size))
    for alpha in range(a_size):
        for T in range(T_size):
            [w, val_Loss[alpha,T]] = linRegression(X_training, y_training, X_validation, y_validation, alphas[alpha], Ts[T])
    # find min combination
    min_index = unravel_index(np.nanargmin(val_Loss), val_Loss.shape)
    alpha_opt = alphas[min_index[0]]
    T_opt = Ts[min_index[1]]
    np.save('hyper_losses', val_Loss)
    #clear data for visualization
    return [val_Loss, alpha_opt, T_opt]

def plotHyperparameters():
    #clear data
    val_Loss = np.load('hyper_losses.npy')
    alphas = np.array([0.1, 0.02, 0.015, 0.012, 0.011, 0.01, 0.009, 0.008, 0.005])
    Ts = np.array([100, 300, 500, 800, 1000, 3000])
    max_Loss = np.nanmax(val_Loss[val_Loss != np.inf])
    val_Loss = np.nan_to_num(val_Loss, copy= False, nan= max_Loss +1, posinf = max_Loss+1)
    
    T_axis, alpha_axis = np.meshgrid(Ts, alphas)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(alpha_axis, T_axis, val_Loss )
    plt.show()
      
def main():
    #2. generate data
    #initialize var. according to assignment
    x_range = np.array([-3, 2])
    w_star = np.array([[-8],[-4],[2],[1]])
    sigma = 0.5
    sample_size = 100
    seed_training = 0
    seed_validation = 1
    [X_training, y_training] = create_dataset(w_star, x_range, sample_size, sigma, seed_training)
    [X_validation, y_validation] = create_dataset(w_star, x_range, sample_size, sigma, seed_validation)

    #3. generate scatterplot
    #x can be found as column of first order monomials
    x_training = X_training[:,1]
    x_validation = X_validation[:,1]
    fig, ax = plt.subplots()
    ax.set_xlabel("x", fontsize = 16)
    ax.set_ylabel("y", fontsize = 16)
    ax.scatter(x_training,y_training, label ='training data')
    ax.scatter(x_validation, y_validation, c = 'C2', label = 'validation data')
    ax.legend()
    ax.grid(True)
    plt.show()
    
    # 4. torch.nn.linear applies a linear transformation y = x^T A + b to incomoing data
    #bias = False: the layer does not learn a bias b
    # It should be set to False because a bias is already learned as offset in the polynomial
    
    #5. linear Regression
    [w_trained, val_loss] = linRegression(X_training, y_training, X_validation, y_validation)
    #6. report learning rate and number of iterations
    [val_Loss, alpha_opt, T_opt] = findHyperparameter(X_training, y_training, X_validation, y_validation)
    print(f"alpha: {alpha_opt}, T: {T_opt}")
    #plotHyperparameters()
    
    
 
if __name__ == '__main__':
    main()
