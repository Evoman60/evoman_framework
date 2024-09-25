# imports framework
import os
import numpy as np
from evoman.environment import Environment
from evoman.controller import Controller
import neat
import pickle
import pygame
import networkx as nx
import matplotlib.pyplot as plt
import json

experiment_name = 'optimization_specialist_neat'

EXPERIMENTS_FILE = 'experiments.json'

def create_experiment_directory(experiment_name):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    if not os.path.exists(f'{experiment_name}/checkpoints'):
        os.makedirs(f'{experiment_name}/checkpoints')

create_experiment_directory(experiment_name)

class NeatPlayerController(Controller):
    def __init__(self):
        self.net = None

    def control(self, inputs, _):
        if self.net is None:
            return [0] * 5  # Default action if network is not set
        # Normalize inputs
        inputs = (inputs - min(inputs)) / (max(inputs) - min(inputs))
        output = self.net.activate(inputs)
        return [int(o > 0.5) for o in output]

    def set(self, net, _):
        self.net = net

player_controller = NeatPlayerController()

def load_experiments():
    try:
        with open(EXPERIMENTS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_experiments(experiments):
    with open(EXPERIMENTS_FILE, 'w') as f:
        json.dump(experiments, f, indent=2)

def choose_experiment():
    global experiment_name
    experiments = load_experiments()
    
    print("\nAvailable experiments:")
    for i, (exp, info) in enumerate(experiments.items(), 1):
        status = "Finished" if info.get('status') == 'finished' else "In progress"
        print(f"{i}. {exp} - Status: {status}, Best Fitness: {info.get('best_fitness', 'N/A')}, Current Generation: {info.get('current_generation', 'N/A')}/{info.get('total_generations', 'N/A')}, Enemy: {info.get('enemy', 'N/A')}")
    print(f"{len(experiments) + 1}. Create new experiment")
    
    choice = input(f"Choose an experiment (1-{len(experiments) + 1}): ").strip()
    
    if choice.isdigit():
        choice = int(choice)
        if 1 <= choice <= len(experiments):
            experiment_name = list(experiments.keys())[choice - 1]
            print(f"Switched to experiment: {experiment_name}")
        elif choice == len(experiments) + 1:
            new_name = input("Enter new experiment name: ").strip()
            if new_name:
                experiment_name = new_name
                experiments[new_name] = {'status': 'in_progress', 'best_fitness': 'N/A', 'current_generation': 0, 'total_generations': 'N/A', 'enemy': 'N/A', 'config_file': 'N/A'}
                save_experiments(experiments)
                create_experiment_directory(experiment_name)
                print(f"Created new experiment: {experiment_name}")
            else:
                print("Invalid experiment name. Keeping current experiment.")
        else:
            print("Invalid choice. Keeping current experiment.")
    else:
        print("Invalid input. Keeping current experiment.")

def update_experiment_info(fitness, current_generation, total_generations, enemy, config_file, isDone = False):
    experiments = load_experiments()
    if experiment_name in experiments:
        experiments[experiment_name]['best_fitness'] = fitness
        experiments[experiment_name]['current_generation'] = current_generation
        experiments[experiment_name]['total_generations'] = total_generations
        experiments[experiment_name]['enemy'] = enemy
        experiments[experiment_name]['config_file'] = os.path.basename(config_file)
        experiments[experiment_name]['status'] = 'finished' if isDone else 'in_progress'
    else:
        experiments[experiment_name] = {
            'status': 'finished' if isDone else 'in_progress',
            'best_fitness': fitness,
            'current_generation': current_generation,
            'total_generations': total_generations,
            'enemy': enemy,
            'config_file': os.path.basename(config_file)
        }
    save_experiments(experiments)

def create_environment(experiment_name, headless, enemy):
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    else:
        os.environ.pop("SDL_VIDEODRIVER", None)
    return Environment(experiment_name=experiment_name,
                       enemies=[enemy],
                       playermode="ai",
                       player_controller=NeatPlayerController(),
                       enemymode="static",
                       level=2,
                       speed="fastest" if headless else "normal",
                       visuals=not headless)

def run_neat(config_file, nb_runs, enemy, checkpoint=None, tot_nb_runs = 0):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    if checkpoint:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
        print(f"Restored from checkpoint: {checkpoint}")
    else:
        pop = neat.Population(config)
    
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(1, filename_prefix=f'{experiment_name}/checkpoints/neat-checkpoint-gen-'))

    def eval_genomes(genomes, config):
        train_env = create_environment(experiment_name, headless=True, enemy=enemy)
        best_fitness = 0
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            f, p, e, t = train_env.play(pcont=net)
            genome.fitness = f
            best_fitness = max(best_fitness, f)
        del train_env
        
        # Update experiment info after each generation
        if checkpoint:
            update_experiment_info(best_fitness, pop.generation, tot_nb_runs, enemy, config_file)
        else:
            update_experiment_info(best_fitness, pop.generation, nb_runs, enemy, config_file)



    winner = pop.run(eval_genomes, nb_runs) 

    # Save the winner
    with open(f'{experiment_name}/winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    print(f'Winner genome saved to {experiment_name}/winner.pkl')

    # Save the best network
    best_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open(f'{experiment_name}/best_net.pkl', 'wb') as f:
        pickle.dump(best_net, f)
    print(f'Best network saved to {experiment_name}/best_net.pkl')

    # Update experiment info
    if checkpoint:
        update_experiment_info(winner.fitness, tot_nb_runs, tot_nb_runs, enemy, config_file, isDone=True)
    else:
        update_experiment_info(winner.fitness, nb_runs, nb_runs, enemy, config_file, isDone=True)

def test_best_network(enemy):
    try:
        with open(f'{experiment_name}/best_net.pkl', 'rb') as f:
            best_net = pickle.load(f)
        print(f'Best network loaded from {experiment_name}/best_net.pkl')
    except FileNotFoundError:
        print("No best network found. Please train the model first.")
        return

    test_env = create_environment(experiment_name, headless=False, enemy=enemy)
    f, p, e, t = test_env.play(pcont=best_net)
    print(f"Fitness: {f}\nPlayer Life: {p}\nEnemy Life: {e}\nTime: {t}")
    del test_env

    pygame.quit()

def show_best_network():
    try:
        with open(f'{experiment_name}/best_net.pkl', 'rb') as f:
            best_net = pickle.load(f)
        print(f'Best network loaded from {experiment_name}/best_net.pkl')
    except FileNotFoundError:
        print("No best network found. Please train the model first.")
        return

    # Create a graph of the best network
    G = nx.DiGraph()
    input_nodes = set(best_net.input_nodes)
    output_nodes = set(best_net.output_nodes)
    hidden_nodes = set(node for node, _, _, _, _, _ in best_net.node_evals) - input_nodes - output_nodes

    for node, _, _, _, _, links in best_net.node_evals:
        G.add_node(node)
        for i, w in links:
            G.add_edge(i, node, weight=round(w,4))

    plt.figure(figsize=(8, 5))
    
    # Define positions for nodes based on their type
    pos = {}
    layer_dist = 1 / (len(hidden_nodes) + 1)
    
    for i, node in enumerate(sorted(input_nodes)):
        pos[node] = (0, i / (len(input_nodes) - 1) if len(input_nodes) > 1 else 0.5)
    
    for i, node in enumerate(sorted(hidden_nodes)):
        pos[node] = (0.5, (i + 1) * layer_dist)
    
    for i, node in enumerate(sorted(output_nodes)):
        pos[node] = (1, i / (len(output_nodes) - 1) if len(output_nodes) > 1 else 0.5)
    
    # Draw nodes with different colors based on their type
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='#76c7c0', node_size=700, label='Input Nodes')
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='#ffcc5c', node_size=700, label='Hidden Nodes')
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='#ff6f69', node_size=700, label='Output Nodes')
    
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='#d3d3d3')
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', font_weight='bold')
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')
    
    plt.title("Visualization of the Best NEAT Network", fontsize=20, fontweight='bold', color='black')
    plt.legend(loc='upper right', fontsize='large')
    plt.gca().set_facecolor('white')
    plt.show()

def interactive_menu():
    global experiment_name

    while True:
        experiments = load_experiments()
        current_exp_info = experiments.get(experiment_name, {})
        print(f"\nNEAT Evoman Framework Menu (Current Experiment: {experiment_name})")
        print(f"Status: {current_exp_info.get('status', 'N/A')}, Best Fitness: {current_exp_info.get('best_fitness', 'N/A')}, Current Generation: {current_exp_info.get('current_generation', 'N/A')}/{current_exp_info.get('total_generations', 'N/A')}, Enemy: {current_exp_info.get('enemy', 'N/A')}, Config File: {current_exp_info.get('config_file', 'N/A')}")
        print("1. Train new model")
        print("2. Continue training from checkpoint")
        print("3. Test best network")
        print("4. Show best network")
        print("5. Change experiment")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            config_file = input("Enter path to the NEAT configuration file (default: neat_config.txt): ").strip() or 'neat_config.txt'
            nb_runs = int(input("Enter number of runs for training (default: 10): ").strip() or '10')
            enemy = int(input("Choose an enemy (1-8): ").strip())
            if 1 <= enemy <= 8:
                checkpoints = [f for f in os.listdir(f'{experiment_name}/checkpoints') if f.startswith('neat-checkpoint-gen-')]
                if checkpoints:
                    for cp in checkpoints:
                        os.remove(f'{experiment_name}/checkpoints/{cp}')
                run_neat(config_file, nb_runs, enemy)
            else:
                print("Invalid enemy number. Please choose a number between 1 and 8.")
        elif choice == '2':
            checkpoints = [f for f in os.listdir(f'{experiment_name}/checkpoints') if f.startswith('neat-checkpoint-gen-')]
            if not checkpoints:
                print("No checkpoints found.")
                continue
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  
            print("Available checkpoints:")
            for i, cp in enumerate(checkpoints, 1):
                print(f"{i}. {cp}")
            cp_choice = int(input("Choose a checkpoint number: "))
            if 1 <= cp_choice <= len(checkpoints):
                checkpoint = os.path.join(f'{experiment_name}/checkpoints', checkpoints[cp_choice-1])
                config_file = current_exp_info.get('config_file', 'neat_config.txt')
                total_generations = current_exp_info.get('total_generations', 10)
                enemy = current_exp_info.get('enemy')

                # Extract the number of completed generations from the checkpoint filename
                completed_generations = int(checkpoint.split('-')[-1])
                remaining_generations = total_generations - completed_generations
                run_neat(config_file, remaining_generations, enemy, checkpoint=checkpoint, tot_nb_runs=total_generations)
            else:
                print("Invalid checkpoint number.")
        elif choice == '3':
            enemy = int(input("Choose an enemy to test against (1-8): ").strip())
            if 1 <= enemy <= 8:
                test_best_network(enemy)
            else:
                print("Invalid enemy number. Please choose a number between 1 and 8.")
        elif choice == '4':
            show_best_network()
        elif choice == '5':
            choose_experiment()
        elif choice == '6':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == '__main__':
    interactive_menu()