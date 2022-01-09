def test_hyperparameters(X_train, y_train, X_valid, y_valid, D, alpha, k, sigma, lbd, epochs):
    model = FM(D, alpha, k, sigma, lbd)
    model.train(X_train, y_train, epochs, compute_loss = False, X_valid = None, y_valid = None, print_evol = False)
    return {'validation_loss': model.compute_validation_loss(X_valid, y_valid), 'training_loss': model.training_loss}

class Grid:

    mapping = {'D':0, 'alpha':1, 'k':2, 'sigma':3, 'lambda':4, 'epochs':5}

    def __init__(self, Ds, alphas, ks, sigmas, lbds, epochs):
        self.Ds = Ds
        self.alphas = alphas
        self.ks = ks
        self.sigmas = sigmas
        self.lbds = lbds
        self.epochs = epochs

        # A dict with the associated losses
        self.loss_dict = {}

        # Create a list of all possible combinations associated with the grid
        self.combinations = []
        for D in self.Ds:
          for alpha in self.alphas:
            for k in self.ks:
              for sigma in self.sigmas:
                for lbd in self.lbds:
                  for iter in self.epochs:
                      self.combinations.append([D,alpha,k,sigma,lbd,iter])


    def grid(self):
        return {'D':self.Ds, 'alpha':self.alphas, 'k':self.ks, 'sigma':self.sigmas, 'lambda':self.lbds, 'epochs':self.epochs}


    def reduce_combinations(self, param1, param2 = None):
        p1 = self.mapping[param1]
        if param2 is not None:
            p2 = self.mapping[param2]

        combinations = self.combinations

        for elem in combinations:
            elem[p1] = None
            if param2 is not None:
                elem[p2] = None
        
        reduced_combinations = []
        for elem in combinations:
            if elem not in reduced_combinations:
                reduced_combinations.append(elem)
                
        return reduced_combinations


    def generate_combination_label(self, combination: list):
        label = ''
        keys = [key for key in self.mapping]
        for i in range(len(combination)):
            if combination[i] is not None:
                label += ', ' + keys[i] + ' = ' + str(combination[i])
        return label[2:]


    # Grid search function
    @exec_time
    def search(self, X_train, y_train, X_valid, y_valid):
        Ds = self.Ds
        alphas = self.alphas
        ks = self.ks
        sigmas = self.sigmas
        lbds = self.lbds
        epochs = self.epochs
        for D in Ds:
          for alpha in alphas:
            for k in ks:
              for sigma in sigmas:
                for lbd in lbds:
                  for iter in epochs:
                      print('D =',D, 'alpha =', alpha, 'k =:', k, 'sigma =', sigma, 'lambda =', lbd, 'epochs =',iter)
                      # The parameters set we test
                      losses = test_hyperparameters(X_train, y_train, X_valid, y_valid, D, alpha, k, sigma, lbd, iter)

                      self.loss_dict[D, alpha, k, sigma, lbd, iter] = [losses['validation_loss'], losses['training_loss']]
        pass

    '''def remove_from_grid_results(self, param: str, value):

        p = self.mapping[param]
        for key in self.loss_dict:
          if key[p] == value:
            self.loss_dict.pop(self.loss_dict[key])
        pass'''


    def plot_grid_search(self, param1: str, param2: str):
      # Plot validation loss for different values of param1 (x-axis) and param2 (different colors) for each possible value of the other parameters (on different plots)
        p1, p2 = self.mapping[param1], self.mapping[param2]

        key_list = [i for i in range(len(self.mapping)) if (i!= p1 and i!= p2)]


        combinations = self.reduce_combinations(param1, param2)

        N = int(sqrt(len(combinations)))+1
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(6*N, 6*N)
        print('Grid search for different values of '+param1+' and '+param2)

        count = 0
        for elem in combinations:

          count += 1
          plt.subplot(N,N,count)
          #print(N,N,count)
          plot_data = dict()

          for key in self.loss_dict:

            if [elem[i] for i in key_list] == [key[i] for i in key_list]:
              if not(key[p2] in plot_data.keys()):
                  plot_data[key[p2]] = [[],[]]
              plot_data[key[p2]][0].append(key[p1])
              plot_data[key[p2]][1].append(self.loss_dict[key][0])

          for key in plot_data:
            title = self.generate_combination_label(elem)
            label = str(key)

            plt.plot(plot_data[key][0],plot_data[key][1], '+-', linewidth=1, markersize=22, label = label)
            plt.title(title)
            plt.xlabel(param1)
            plt.ylabel('logloss')
            leg = plt.legend(loc='best')
        plt.show()

    def plot_losses(self, param: str, training = False, validation = True):
      # Plot training loss vs validation loss for different values of the selected 'param' for each possible value of the other parameters in the grid
        pass
