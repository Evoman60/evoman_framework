# the demo_controller file contains standard controller structures for the agents.
# you can overwrite the method 'control' in your own instance of the environment
# and then use a different type of controller if you wish.
# note that the param 'controller' received by 'control' is provided through environment.play(pcont=x)
# 'controller' could contain either weights to be used in the standard controller (or other controller implemented),
# or even a full network structure (ex.: from NEAT).
from evoman.controller import Controller
import numpy as np
from nn import NN, MLP

def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))

# implements controller structure for player
class player_controller(Controller):
    def __init__(self, controller):    
        self.controller = controller
    def set(self,controller, n_inputs):
        # Number of hidden neurons
        if isinstance(controller, NN):
            self.controller = controller
        # n_inputs and this elif are only there for potential compatibility reasons
        elif isinstance(controller, (np.ndarray, str)): 
            assert n_inputs == self.controller.input_size
            self.controller.set(controller)
        else:
            raise NotImplementedError

    def control(self, inputs, controller):
        self.controller.set(controller)
        output = self.controller.forward(inputs) 
        threshhold = self.controller.activation_function(0)
        
        if output[0] > threshhold:
            left = 1
        else:
            left = 0

        if output[1] > threshhold:
            right = 1
        else:
            right = 0

        if output[2] > threshhold:
            jump = 1
        else:
            jump = 0

        if output[3] > threshhold:
            shoot = 1
        else:
            shoot = 0

        if output[4] > threshhold:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]


# implements controller structure for enemy
class enemy_controller(Controller):
    def __init__(self, _n_hidden):
        # Number of hidden neurons
        self.n_hidden = _n_hidden

    def control(self, inputs, controller):
        # Normalises the input using min-max scaling
        #why initialize it here
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
        self.controller = MLP(len(inputs), 1, self.n_hidden, 6, controller)
        output = self.controller.forward(inputs)
        threshhold = self.controller.activation_function(0)
        
        if output[0] > threshhold:
            attack1 = 1
        else:
            attack1 = 0

        if output[1] > threshhold:
            attack2 = 1
        else:
            attack2 = 0

        if output[2] > threshhold:
            attack3 = 1
        else:
            attack3 = 0

        if output[3] > threshhold:
            attack4 = 1
        else:
            attack4 = 0
            
        if output[4] > threshhold:
            attack5 = 1
        else:
            attack5 = 0
        
        if output[5] > threshhold:
            attack6 = 1
        else:
            attack6 = 0
            
        return [attack1, attack2, attack3, attack4, attack5, attack6]
