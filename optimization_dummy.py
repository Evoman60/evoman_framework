###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from customed_controller import player_controller

# imports other libs
import numpy as np
import os

# runs simulation
def simulation(env:Environment, x: np.ndarray):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation is a batch version of simulation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))



def main(dev_mode:str = 'visual_debug', n_hidden_neurons:int =10) -> None:
    
    if dev_mode == 'visual_debug':
        visuals=True
        speed='normal'
    else:
        os.environ["SDL_VIDEODRIVER"]="dummy" # choose this for not using visuals and thus making experiments faster
        visuals=False
        speed='fastest'


    experiment_name = 'dummy_demo' + f'_{dev_mode}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="normal",
                    fullscreen=False,
                    visuals=visuals)


    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    
    # start writing your own code from here
    


if __name__ == '__main__':
    main()