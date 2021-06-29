import torch
import torch.nn as nn
import torch.optim as opti
import pickle
import numpy as np
import pandas as pd
import random
import csv
from sklearn.metrics import mean_squared_error as mse

class network(nn.Module):

    def __init__(self, input_dim, neurons):
        super(network, self).__init__()

        self.input_dim = input_dim
        self.neurons = neurons
        self.layers = []
        self.layers.append(nn.Linear(input_dim, neurons[0]))

        for i in range(len(neurons)):
            if(i != len(neurons)-1):
                self.layers.append(nn.ReLU())
            if(i != len(neurons)-1):
                self.layers.append(nn.Linear(neurons[i], neurons[i+1]))

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        
        x = self.net(x)

        return x

    
class Regressor():

    def __init__(self, x, nb_epoch = 765, learning_rate=0.09002852101468667, neurons=[31, 28, 32, 21, 1]
, batch_size=410):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        
        self.lossLayer = nn.MSELoss()   
        self.learning_rate = learning_rate
        self.neurons = neurons
        self.min = 0
        self.max = 0
        self.width = 0
        self.batch_size = batch_size
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        
        self.model = network(self.input_size, self.neurons)

        self.output_size = 1
        self.nb_epoch = nb_epoch 
        


        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        
        onehot = pd.get_dummies(x['ocean_proximity'])
        x = pd.concat([x, onehot], axis = 1)
        x.drop(columns = 'ocean_proximity', axis = 1, inplace = True) 
        if(training):
            #store min max and avg.
            self.avg = x.mean()
            self.min = np.min(x, axis = 0)
            self.max = np.max(x, axis = 0)
        
        x.fillna(self.avg, inplace = True)

        normalised_input = (x-self.min)/(self.max-self.min) #normalise between 0 and 1


        return torch.tensor(normalised_input.to_numpy()), (torch.tensor(y.to_numpy()) if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        optimiser = opti.Adam(self.model.parameters(), lr=self.learning_rate)
        for i in range (self.nb_epoch):
            #batches
            indices = np.random.permutation(len(X))
            shuffled_i = X[indices]
            shuffled_t = Y[indices]

            batch_i = torch.split(shuffled_i, self.batch_size)
            batch_t = torch.split(shuffled_t, self.batch_size)


            for y in range(len(batch_i)): #iterate over batches

                output = self.model(batch_i[y].float())
                loss = self.lossLayer(output, batch_t[y].float())
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()


        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        prediction = self.model(X.float())

        return prediction.detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        prediction = self.predict(x)

        return mse(Y.detach().numpy(), prediction, squared=False) # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x,y,xv,yv): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    fields = ["Learning Rate", "Epochs", "Layers", "Batch Size", "RMSE", ]
    rmse = np.inf
    best_model = {}
    with open('data.csv', mode='w') as data:
        writer = csv.writer(data)
        writer.writerow(fields)
        for i in range(100):
            epoch = random.randint(300,1000)
            batch_sz = random.randint(250,500)
            neuron_no = random.sample(range(10,40), random.randint(3,7))
            neuron_no[-1] = 1
            lr = random.uniform(0.07,0.1)
            # print("=================================================\n")
            # print('Epoch: ', epoch)
            # print('Batch Size: ', batch_sz)
            # print('Neurons:', neuron_no)
            # print('Learning Rate: ', lr)

            tuning_model = Regressor(x, nb_epoch = epoch, learning_rate= lr, neurons = neuron_no , batch_size = batch_sz )
            tuning_model.fit(x,y)
            prediction = tuning_model.predict(x)
            score = tuning_model.score(xv,yv)

            params = [lr, epoch, neuron_no, batch_sz, score]
            
            writer.writerow(params)

            # print(score,'\n')

            if(score < rmse):
                rmse = score
                save_regressor(tuning_model)
                best_model = {"epoch":epoch, "batch_size":batch_sz, "Neurons": neuron_no, "learning rate": lr, "score":score}

    return  best_model

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

    

def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # x_trainer = x_train.sample(frac=0.8,random_state=200) #random state is a seed value
    # y_trainer = y_train.sample(frac=0.8,random_state=200) #random state is a seed value

    # x_val = x_train.drop(x_trainer.index)
    # y_val = y_train.drop(y_trainer.index) 

    # x_test = x_val.sample(frac=0.5, random_state=2000)
    # y_test = y_val.sample(frac=0.5, random_state=2000)

    # x_val = x_val.drop(x_test.index)
    # y_val = y_val.drop(y_test.index)


    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
 

    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    #save_regressor(regressor)

    #print(RegressorHyperParameterSearch(x_trainer, y_trainer, x_val, y_val )) #hyperparameter tuning
    #r = load_regressor()
    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))




if __name__ == "__main__":
    example_main()
    
