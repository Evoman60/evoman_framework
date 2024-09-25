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

def parent_selection(population, fitness, parent_num, mode: str, **kwarg) -> np.ndarray:
    probility = minsum_norm(fitness)
    if mode == 'fitness_propotional_selection':
        selected_index = np.random.choice(population.shape[0], parent_num, True, probility)
        parents = population[selected_index]
    elif mode == 'tournament':
        candidate_index = np.random.choice(population.shape[0], kwarg['parent_tournament_size'])
        selected_index = fitness[candidate_index].argpartition(parent_num)[-parent_num]
        parents = population[selected_index]
    return parents

def crossover(population, 
              fittness,
              parent_num : int = 2, 
              offspring_num : int | None | str = None,
              parent_selection_mode: str = 'fitness_propotional_selection', 
              crossover_mode: str = 'whole_arithmetic',
              alpha : np.ndarray = np.array([0.5, 0.5]),
              **kwarg):
    
    #alpha is a parameter to control the propotion to blend in various crossover strategy.
    #alpha is a list. each element in the list specifies the propotion for each parents' to the offspring. 
    
    offspring_list = []
    while len(offspring_list) < offspring_num:  
        parent_array = parent_selection(population, fittness, parent_num, parent_selection_mode, **kwarg)
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



def population_selection(population, fitness: np.ndarray, population_num, mode: str, elite_num = 3):
    if population.shape[0] < population_num:
        return population, fitness
    probility = minsum_norm(fitness)
    if 'fitness_propotional_selection' in mode:

        selected_index = np.random.choice(population.shape[0], population_num, True, probility)
        if 'elitism' in mode:
            best_index = fitness.argpartition(elite_num)[-elite_num : ]
            for i in best_index.tolist():
                if i not in selected_index:
                    selected_index[np.random.randint(0, selected_index.shape[0])] = i
        population = population[selected_index]
        fitness = fitness[selected_index]
        return population, fitness


def train(experiment_name: str, 
          env: Environment, 
          generation_num: int, 
          population_num: int, 
          population_selection_mode: str,
          crossover_mode: str,
          parent_num: int,
          parent_selection_mode: str,
          offspring_num: int,
          alpha: float,
          mutation_mode: str,
          mutation_rate: float,
          **kwarg):
    
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
        offsprings = crossover(population, population_fitness, parent_num, offspring_num, parent_selection_mode, crossover_mode, alpha, **kwarg)
        offsprings = mutation(offsprings, mutation_rate, mutation_mode, **kwarg)
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
        print( f'\n GENERATION {i} {round(population_fitness[best],6)} {round(mean,6)} {round(std,6)}')
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




def main(run_mode: str = 'visual_debug', 
         generation_num: int = None, 
         population_num: int = None,
         crossover_mode: str = None, 
         parent_num: int = None,
         parent_selection_mode: str = None,
         offspring_num: int = None,
         alpha: np.ndarray = None,
         mutation_mode: str = 'guassian', 
         mutation_rate: float = None, 
         population_selection_mode: str = None,
         train_experiment_name: str = None,
         enemies: list = [1],
         activation_function_name = 'sigmoid',
         **kwargs) -> None:
    
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
    
    
    experiment_name = f'dummy_demo_{run_mode}_{population_num}_{crossover_mode}_{parent_selection_mode}_{parent_num}_{offspring_num}_{mutation_mode}_{mutation_rate}_{population_selection_mode}_{alpha.tolist()}_{enemies}_{kwargs.items()}_{activation_function_name}' 
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    print(experiment_name)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=enemies,
                    playermode="ai",
                    player_controller=Controller(), #use a empty one to enable sensor
                    enemymode="static",
                    level=2,
                    speed=speed,
                    fullscreen=False,
                    visuals=visuals)

    sensor_num = env.get_num_sensors()
    new_player_controller = player_controller(MLP(sensor_num, 1, 10, 5,'random', activation_function_name)) #real play controller
    env.player_controller =  new_player_controller
    
    #var_num = len(player_controller.controller.get_weights_as_vec())
    
    # start writing your own code from here
    env.state_to_log()
    
    if 'train' in run_mode or 'debug' in run_mode:
        train(experiment_name=experiment_name,
              env=env,
              generation_num=generation_num,
              population_num=population_num,
              population_selection_mode=population_selection_mode,
              crossover_mode = crossover_mode,
              parent_num=parent_num,
              parent_selection_mode=parent_selection_mode,
              offspring_num=offspring_num,
              alpha = alpha,
              mutation_mode=mutation_mode,
              mutation_rate=mutation_rate,
              **kwargs)
    # initializes population loading old solutions or generating new ones
   
   
    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

if __name__ == '__main__':
    np.random.seed(42)
    
    parent_num=2
    
    main(run_mode = 'debug', 
         generation_num=100, 
         population_num=100, 
         crossover_mode = 'whole_arithmetic',
         parent_num=2,
         parent_selection_mode='tournament',
         population_selection_mode='fitness_propotional_selection_elitism+',
         offspring_num = 100, 
         mutation_mode = 'guassian', 
         mutation_rate = 0.2,
         alpha=np.array([0.5, 0.5]),
         train_experiment_name=None,
         activation_function_name='tanh',
         enemies=[2],
         loc = 0,
         scale = 1,
         parent_tournament_size = parent_num * 5)
    