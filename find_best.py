import os
import shutil
import numpy as np

# Create the destination folder if it doesn't exist
if not os.path.exists('bests'):
    os.makedirs('bests')

# Iterate over the parameters
for parent_num in [2, 3]:
    for enemies in [[2], [3], [4]]:
        max_value = -np.inf
        best_run = None
        
        for run in range(10):
            alpha = np.array([1.0 / parent_num] * parent_num)
            parent_tournament_size = parent_num * 3
            folder_name = (
                f"dummy_demo_debug_100_whole_arithmetic_tournament_{parent_num}_100_guassian_0.2_fitness_propotional_selection_elitism+_"
                f"{alpha.round(2).tolist()}_{enemies}_[0, 1, 1, -1, {parent_tournament_size}, {run}]_relu"
            )
            
            # Read the results.txt file
            results_file_path = os.path.join(folder_name, 'results.txt')
            with open(results_file_path, 'r') as f:
                    for line in f:
                        if len(line.split()) != 0:
                            if line.split()[0][0].isdigit():
                                numbers = list(map(float, line.split()))
                                max_value_in_line = numbers[1]
                                if max_value_in_line > max_value:
                                    max_value = max_value_in_line
                                    best_run = run
        
        # Copy the best.txt file to the bests folder with a clear name
        if best_run is not None:
            best_folder_name = (
                f"dummy_demo_debug_100_whole_arithmetic_tournament_{parent_num}_100_guassian_0.2_fitness_propotional_selection_elitism+_"
                f"{alpha.round(2).tolist()}_{enemies}_[0, 1, 1, -1, {parent_tournament_size}, {best_run}]_relu"
            )
            best_file_path = os.path.join(best_folder_name, 'best.txt')
            if os.path.exists(best_file_path):
                print(1111111)
                destination_file_name = f"best_parent_{parent_num}_enemy_{enemies}_run_{best_run}_{max_value}.txt"
                destination_file_path = os.path.join('bests', destination_file_name)
                shutil.copy(best_file_path, destination_file_path)

            
 