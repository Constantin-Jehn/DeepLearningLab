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


def create_dataset(w_star, x_range, sample_size, sigma, seed= None, degree = 3):
    random_state = np.random.RandomState(seed)
    #array of random numbers (drawn from unifrom distr) of lingth sample size
    x = random_state.uniform(x_range[0], x_range[1], (sample_size))
    X = np.zeros((sample_size, degree + 1))
    for i in range(sample_size):
        X[i,0] = 1
        for j in range(1, degree + 1):
            X[i,j] = x[i]**j
    #use only the underlying polynomial to get targets
    X_underlying = X[:,0:4]
    y = X_underlying.dot(w_star)
    if sigma > 0:
        y += random_state.normal(0.0, sigma, sample_size).reshape(-1,1)
    return X,y

def generateData(x_range = np.array([-3, 2]), w_star = np.array([[-8],[-4],[2],[1]]), sigma = 0.5, training_sample_size = 100, validation_sample_size = 100, seed_training = 0, seed_validation = 1, degree = 3):
    [X_training, y_training] = create_dataset(w_star, x_range, training_sample_size, sigma, seed_training, degree = degree)
    [X_validation, y_validation] = create_dataset(w_star, x_range, validation_sample_size, sigma, seed_validation, degree = degree)
    return [X_training, y_training, X_validation, y_validation]
        
def linRegression(X_tr, y_tr, X_val, y_val, alpha = 0.011, T = 1000, plot = False, degree = 3):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #coefficients to train
    w_star = np.zeros((degree + 1,1))
    #learning rate
    #alpha = 0.011
    #training steps
    #T = 800
    #feature dimension
    in_features = degree + 1
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
    
    if plot:
        train_Loss_plot = np.zeros(T)
        val_Loss_plot = np.zeros(T)
    
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
        if plot:
            train_Loss_plot[step] = loss.item()
            val_Loss_plot[step] = val_loss.item()
    w_star = model.weight
    if plot:
        T_axis = np.linspace(0,T-1,T)
        fig, ax = plt.subplots()
        ax.plot(T_axis, train_Loss_plot, label = 'Training loss')
        ax.plot(T_axis, val_Loss_plot, label = 'Validation loss')
        ax.legend()
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training set size: {} Validation set size{}".format(X_tr.shape[0],X_val.shape[0]))
        plt.show()
    
    return [w_star, val_loss] 

def findHyperparameter(X_training, y_training, X_validation, y_validation):
    #after some simple test with I realize the learning rate should be around 0.01
    alphas = np.array([0.011, 0.01, 0.009, 0.008, 0.005])
    a_size = np.size(alphas)
    Ts = np.array([100, 300, 500, 800, 1000, 2000, 3000])
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
    alphas = np.array([0.011, 0.01, 0.009, 0.008, 0.005])
    Ts = np.array([100, 300, 500, 800, 1000, 2000, 3000])
    max_Loss = np.nanmax(val_Loss[val_Loss != np.inf])
    val_Loss = np.nan_to_num(val_Loss, copy= False, nan= max_Loss +1, posinf = max_Loss+1)
    
    T_axis, alpha_axis = np.meshgrid(Ts, alphas)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(alpha_axis, T_axis, val_Loss )
    ax.set_title("Hyperparameter study")
    ax.set_xlabel("alpha", fontsize = 12)
    ax.set_ylabel("T", fontsize = 12)
    ax.set_zlabel("Validation loss", fontsize = 12)
    plt.show()

def plotPolynomials(w_star, w_hat):
    x = np.linspace(-5,5,100)
    y_star = [np.polyval(w_star,i) for i in x]
    y_hat = [np.polyval(w_hat,i) for i in x]
    fig, ax = plt.subplots()
    ax.plot(x,y_star, label='Ground truth')
    ax.plot(x, y_hat, label ='Estimated polynomial')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('Comparison ground truth end estimated polynomial')
    plt.show()

def scatterPlot(X_training, y_training, X_validation, y_validation):
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
   
def differTrainingSize():
    sizes = np.array([50,10,5])
    for s in range(3):
        [X_training, y_training, X_validation, y_validation] = generateData(training_sample_size=sizes[s])
        scatterPlot(X_training, y_training, X_validation, y_validation)
        linRegression(X_training, y_training, X_validation, y_validation, plot = True, T = 1500)
        
def differSigma():
    sigmas = [2,4,8]
    for i in range(3):
        [X_training, y_training, X_validation, y_validation] = generateData(sigma = sigmas[i])
        scatterPlot(X_training, y_training, X_validation, y_validation)
        linRegression(X_training, y_training, X_validation, y_validation, plot = True)

def compareOrders():
    w_star = np.array([[-8],[-4],[2],[1]])
    #degree 3
    [X_training, y_training, X_validation, y_validation] = generateData(training_sample_size=10)
    [w_3, val_Loss_3] = linRegression(X_training, y_training, X_validation, y_validation, plot = True)
    print(f"validation loss 3rd order: {val_Loss_3}")
    w_3 = np.transpose(w_3.detach().numpy())
    plotPolynomials(w_star, w_3)
        
    #degree 4: 
    [X_training, y_training, X_validation, y_validation] = generateData(training_sample_size=10, degree = 4)
    [w_4, val_Loss_4] = linRegression(X_training, y_training, X_validation, y_validation, plot = True, degree = 4)
    w_4 = np.transpose(w_4.detach().numpy())
    plotPolynomials(w_star, w_4)
    print(f"validation loss 4th order: {val_Loss_4}")
    
def main():
    #2. generate data
    #initialize var. according to assignment
    
    [X_training, y_training, X_validation, y_validation] = generateData()

    #3. generate scatterplot
    #x can be found as column of first order monomials
    scatterPlot(X_training, y_training, X_validation, y_validation)
    
    # 4. torch.nn.linear applies a linear transformation y = x^T A + b to incomoing data
    #bias = False: the layer does not learn a bias b
    # It should be set to False because a bias is already learned as offset in the polynomial
    
    #5. linear Regression
    [w_trained, val_loss] = linRegression(X_training, y_training, X_validation, y_validation)
    #6. report learning rate and number of iterations
    [val_Loss, alpha_opt, T_opt] = findHyperparameter(X_training, y_training, X_validation, y_validation)
    print(f"alpha: {alpha_opt}, T: {T_opt}")
    plotHyperparameters()
    # T = 1000 and alpha = 0.011 has no significantly higher val loss than the optimimum it's in the abs range of e-5
    # 7. Plot the training and validation loss
    linRegression(X_training, y_training, X_validation, y_validation, plot = True)
    #8. plot the polynomial
    w_star = np.array([[-8],[-4],[2],[1]])
    w_trained = np.transpose(w_trained.detach().numpy())
    plotPolynomials(w_star, w_trained)
    #9. Report on different sized training sets with same size validation set
    differTrainingSize()
    #validation error grows in particular for n = 5 --> heavy overfitting
    #10. differ Sigma 
    differSigma()
    #11. reduce to 10 test observations and compare to 3rd order to 4th order polynomial
    compareOrders()
    
if __name__ == '__main__':
    main()
