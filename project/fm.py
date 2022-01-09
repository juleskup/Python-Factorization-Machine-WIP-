from datetime import datetime
import time
from csv import DictReader
from math import exp, log, sqrt
import numpy as np
import matplotlib.pyplot as plt

class FM:

############################### Init ##############################################################
    def __init__(self, D, alpha, k, sigma, lbd):
        self.D = D
        self.alpha = alpha
        self.k = k
        self.sigma = sigma
        self.lbd = lbd
        
        # initialize our model
        self.w = [0.] * D  # first order parameters
        self.V = [[np.random.normal(0,sigma) for i in range(k)] for j in range(D)] # second order parameters
        self.n = [0.] * D  # number of times we've encountered a feature
        self.N = [[0.]*k]*D

        # Trained booelan
        self.trained = False

        # Losses
        self.training_losses = [] 
        self.validation_losses = [] 

        # Final training loss
        self.training_loss = 0
    
    def hyperparameters(self):
        return {'D':self.D,
                'alpha':self.alpha,
                'k':self.k,
                'sigma':self.sigma,
                'lambda':self.lbd}
    def weights(self):
        return {'w':self.w,
                'V':self.V,
                'n':self.n,
                'N':self.N}

############################### Functions ##############################################################

    # Hashtrick
    # INPUT:
    #     csv_row: a csv dictionary, ex: {'hash_1': '357', 'hash_2': '', ...}
    #     D: the max index that we can hash to
    # OUTPUT:
    #     x: a list of indices that its value is 1
    def get_x(self, csv_row):
        x = []
        for key, value in csv_row.items():
            index = int(value + key[4:], 16) % self.D
            x.append(index)
        return x  # x contains indices of features that have a value of 1


    # Get probability estimation on x
    # INPUT:
    #     x: features
    #     w: weights
    # OUTPUT:
    #     probability of p(y = 1 | x; w)
    # We add the second order term to wTx
    # V is a matrix with n (number of parameters) columns and k rows (coefficients)
    def get_p(self, x):
        wTx = 0.
        for i in x:  # do wTx
            wTx += self.w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
            for j in x:
              if j > i:
                wTx += dot(self.V[i],self.V[j]) # <V[i],V[j]> * x[i] * x[j] as if 'i in x' and 'j in x', we have x[i] = x[j] = 1
        return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid

    # Update given model
    # INPUT:
    #     w: weights
    #     V: second order parameters
    #     n, N: counters that count the number of times we encounter a feature
    #        this is used for adaptive learning rate
    #     x: feature
    #     p: prediction of our model
    #     y: answer
    # OUTPUT:
    #     w: updated model
    #     V: updated model
    #     n: updated count
    #     N: updated count
    def update_wV(self, x, p, y):
        for i in x:
  
            # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
            # (p - y) * x[i] is the current gradient
            # note that in our case, if i in x then x[i] = 1
            self.w[i] -= (p - y) * (1 + 2*self.lbd* self.w[i]) * self.alpha / (sqrt(self.n[i]) + 1.)
            self.n[i] += 1.
            for f in range(self.k):
              sum = - self.V[i][f]
              for j in x:
                sum += self.V[j][f]
              self.V[i][f] -= 2 * (p - y) * (sum + 2*self.lbd* self.V[i][f]) * self.alpha / (sqrt(self.N[i][f]) + 1.)
              self.N[i][f] += 1.
        pass

    

    # Since this always the same data for the validation loss  we  should pin them to memory to iterate faster
    def compute_validation_loss(self, X_valid, y_valid):

        w, V = self.w, self.V

        val_loss = 0 
        for t, (row, y)  in enumerate(zip(DictReader(open(X_valid)), DictReader(open(y_valid)))):
          x = self.get_x(row)
          p = self.get_p(x)
          target = float(y['click'])
          val_loss += logloss(p, target)
        return val_loss/t


################################### Train #########################################################################
    @exec_time
    def train(self, X_train, y_train, epochs, compute_loss = False, X_valid = None, y_valid = None, print_evol = True):
        D = self.D 
        alpha = self.alpha 
        k = self.k 
        sigma = self.sigma 
        lbd = self.lbd 

        n_updates = 0
        training_losses = [] 
        validation_losses = [] 

        if self.trained:
            if prompt('Train again?') == False:
              # re-initialize our model
              self.w = [0.] * D  # first order parameters
              self.V = [[np.random.normal(0,sigma) for i in range(k)] for j in range(D)] # second order parameters
              self.n = [0.] * D  # number of times we've encountered a feature
              self.N = [[0.]*k]*D     


        Hy = getHy(y_train)

        # start training a factorization machine using on pass sgd
        for e in range(epochs):
            training_loss = 0
            for t, (row, y)  in enumerate(zip(DictReader(open(X_train)), DictReader(open(y_train)))):
                # main training procedure
                # step 1, get the hashed features
                x = self.get_x(row)

                # step 2, get prediction
                p = self.get_p(x)
                target = float(y['click'])

                # for progress validation, useless for learning our model, use the optional parameter 'compute_loss' to print the loss 
                training_loss += logloss(p, target)
                if compute_loss and n_updates% 10000 == 0 and n_updates>1:
                        training_losses.append( training_loss/t )
                        validation_losses.append( self.compute_validation_loss(X_valid, y_valid) )
                        print('%s\tupdates: %d\tcurrent logloss on train: %f\tcurrent logloss on validation: %f \tNCE in validation %f' % (
                              datetime.now(), n_updates, training_losses[-1], validation_losses[-1], (Hy-validation_losses[-1])/Hy ))
                elif print_evol and not(compute_loss) and n_updates% 10000 == 0 and n_updates>1:
                        training_losses.append( training_loss/t )
                        print('%s\tupdates: %d' % (datetime.now(), n_updates))

                # step 3, update model with answer
                self.update_wV(x, p, target)

                n_updates += 1

        self.trained = True
        self.training_losses = training_losses
        self.validation_losses = validation_losses

        self.training_loss = training_loss
        pass

################################### Plot #########################################################################

    def plot_losses(self):
      if self.trained:
          x = [10000*i for i in range(len(self.training_losses))]
          plt.plot(x, self.training_losses, label='Train')
          plt.plot(x, self.validation_losses, label='Validation')
          plt.xlabel('Number of updates')
          plt.ylabel('Log Loss')
          plt.legend( ('Train', 'Validation') )
          plt.show()
          return
      return 'Model has not been trained'
