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
from evoman.controller import Controller

# imports other libs
import numpy as np
import os
import time
from nn import NN, MLP

def minsum_norm(arr):
    return (arr - arr.min()) / (arr - arr.min()).sum()


# runs simulation
def simulation(env:Environment, x: np.ndarray):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation is a batch version of simulation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def parent_selection(population, fitness, parent_num, mode: str) -> np.ndarray:
    probility = minsum_norm(fitness)
    if mode == 'fitness_propotional_selection':
        selected_index = np.random.choice(population.shape[0], parent_num, True, probility)
        offsprings = population[selected_index, : ]
        return offsprings

def crossover(population, 
              fittness,
              parent_num : int = 2, 
              offspring_num : int | None | str = None,
              parent_selection_mode: str = 'fitness_propotional_selection', 
              crossover_mode: str = 'whole_arithmetic',
              alpha : np.ndarray = np.array([0.5, 0.5])):
    
    #alpha is a parameter to control the propotion to blend in various crossover strategy.
    #alpha is a list. each element in the list specifies the propotion for each parents' to the offspring. 
    
    offspring_list = []
    while len(offspring_list) < offspring_num:  
        parent_array = parent_selection(population, fittness, parent_num, parent_selection_mode)
        if crossover_mode == 'whole_arithmetic':
            if alpha.sum() != 1.0:
                print(f"WARNING: Using an alpha whose sum is not 1. Currnet alpha sum is {alpha}.")
            offspring = (parent_array * alpha[ : , None]).sum(axis = 0)
            offspring_list.append(offspring)
    
    return np.array(offspring_list)
            
    
    

def mutation(population, mutation_rate, mode: str, **kwarg):
    
    mutation_rate = mutation_rate
    
    if mode == 'identical':
        return population
    
    if mode == "uniform":
        upper_bound, lower_bound = kwarg['upper_bound'], kwarg['lower_bound']
        mutation_operator = lambda gene: np.random.uniform(upper_bound, lower_bound)
    elif mode == "guassian":
        mutation_operator = lambda gene: gene + np.random.normal(kwarg['loc'], kwarg['scale'])
    
    for individual_index in range(population.shape[0]):
        for gene_index in range(population.shape[1]):
            if np.random.rand() < mutation_rate:
                population[individual_index][gene_index] = mutation_operator(population[individual_index][gene_index])            
    
    return population



def population_selection(population, fitness, population_num, mode: str):
    if population.shape[0] < population_num:
        return population, fitness
    probility = minsum_norm(fitness)
    if 'fitness_propotional_selection' in mode:
        selected_index = np.random.choice(population.shape[0], population_num, True, probility)
        population = population[selected_index]
        fitness = fitness[selected_index]
        return population, fitness


def train(experiment_name: str, 
          env: Environment, 
          generation_num: int, 
          population_num: int, 
          population_selection_mode: str,
          mutation_mode: str,
          mutation_rate: float):
    if not os.path.exists(experiment_name+'/evoman_solstate'):

        print( '\nNEW EVOLUTION\n')
        
        # for the reason we use heuristic initialization, the way we initialize is binded with shape of a neural network
        population = np.array([MLP.like(env.player_controller.controller).get_weights_as_vec() for _ in range(population_num)])
        population_fitness = evaluate(env, population)
        best = np.argmax(population_fitness)
        mean = np.mean(population_fitness)
        std = np.std(population_fitness)
        ini_g = 0
        solutions = [population, population_fitness]
        env.update_solutions(solutions)

    else:

        print( '\nCONTINUING EVOLUTION\n')

        env.load_state()
        population = env.solutions[0]
        population_fitness = env.solutions[1]

        best = np.argmax(population_fitness)
        mean = np.mean(population_fitness)
        std = np.std(population_fitness)

        # finds last generation number
        file_aux  = open(experiment_name+'/gen.txt','r')
        ini_g = int(file_aux.readline())
        file_aux.close()
        
    # saves results for first pop
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')
    print( '\n GENERATION '+str(ini_g)+' '+str(round(population_fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(ini_g)+' '+str(round(population_fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()
    
    for i in range(generation_num):
        offsprings = crossover(population, population_fitness, 2, population_num)
        offsprings = mutation(offsprings, mutation_rate, mutation_mode, loc = 0, scale = 1)
        offspring_fittness = evaluate(env, offsprings)
        if "+" in population_selection_mode:
            offsprings = np.concatenate((offsprings, population))
            offspring_fittness = np.concatenate((offspring_fittness, population_fitness))
        population, population_fitness = population_selection(offsprings, offspring_fittness, population_num, population_selection_mode)
        
        best = np.argmax(population_fitness)
        mean = np.mean(population_fitness)
        std = np.std(population_fitness)
        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(population_fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(i)+' '+str(round(population_fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt',population[best])

        # saves simulation state
        solutions = [population, population_fitness]
        env.update_solutions(solutions)
        env.save_state()
        

    env.state_to_log() # checks environment state





def main(run_mode:str = 'visual_debug', generation_num = None, population_num=None, mutation_mode = 'guassian', mutation_rate=None, train_experiment_name = None) -> None:
    
    ini = time.time()
    
    
    if run_mode == 'visual_debug':
        visuals=True
        speed='normal'
        generation_num = 5 if generation_num is None else generation_num
    elif run_mode == 'debug':
        visuals=False
        speed='fastest'
        os.environ["SDL_VIDEODRIVER"]="dummy"
        generation_num = 5 if generation_num is None else generation_num
    elif run_mode =='test':
        #have to specify which experiment to test, for logging reasons, use the different experiment name
        bsol = np.loadtxt(train_experiment_name+'/best.txt') 
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        evaluate([bsol])
    elif run_mode == 'visual_test':
        visuals=True
        speed='normal'
    elif run_mode == 'fast_test':
        os.environ["SDL_VIDEODRIVER"]="dummy" # choose this for not using visuals and thus making experiments faster
        visuals=False
        speed='fastest'
    else:
        os.environ["SDL_VIDEODRIVER"]="dummy" 
        visuals=False
        speed='fastest'
    
    experiment_name = 'dummy_demo' + f'_{run_mode}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=Controller(), #use a empty one to enable sensor
                    enemymode="static",
                    level=2,
                    speed=speed,
                    fullscreen=False,
                    visuals=visuals)

    sensor_num = env.get_num_sensors()
    new_player_controller = player_controller(MLP(sensor_num, 1, 10, 5)) #real play controller
    env.player_controller =  new_player_controller
    
    #var_num = len(player_controller.controller.get_weights_as_vec())
    
    # start writing your own code from here
    env.state_to_log()
    
    if 'train' in run_mode or 'debug' in run_mode:
        train(experiment_name=experiment_name,
              env=env,
              generation_num=generation_num,
              population_num=population_num,
              population_selection_mode='fitness_propotional_selection+',
              mutation_mode=mutation_mode,
              mutation_rate=mutation_rate)
    # initializes population loading old solutions or generating new ones
   
   
    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()
   
if __name__ == '__main__':
    main('visual_debug', generation_num=100, population_num=10, mutation_rate=0.2)
    