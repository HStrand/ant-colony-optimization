import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
from aco.colony import *
from aco.ant import *

"""
Import data
"""
distances = pd.read_csv('data/distances.csv', sep=';') # Get distances from file
numerated = {}
cities = distances.columns.values
for i in range(1,len(cities)):
    numerated[i] = cities[i]

INVERSE_IDENTITY = 1 - np.identity(len(distances)) # Inverse identity matrix

"""
Helper functions
"""
def get_row_by_num(num):
	return np.array(distances)[num,1:]

def get_distance(a, b):
	return get_row_by_num(a)[b]

def get_city_name(x):
    return numerated[x]

def generate_matrix(initial_pheromone):
	P = np.array([[initial_pheromone]*len(distances)]*len(distances)) # Initial transition matrix	
	P = P * INVERSE_IDENTITY # Set diagonal to 0 to avoid transitionss to current state
	return P

"""
Brute-force search algorithm
"""
def brute_force(initial_location):
	cities = []
	for i in range(0,len(distances)):
		if not (i==initial_location):
			cities.append(i)

	print("---------------------------")
	print("Starting brute-force search")
	start = time.time()
	shortest_path_distance = 999999999
	shortest_path = []
	combos = itertools.permutations(cities,len(cities))
	i = 0
	for r in combos:
	    distance = 0    
	    distance += get_distance(initial_location,r[0])
	    
	    for i in range(0,len(r)-1):
	        distance += get_distance(r[i],r[i+1])        
	        
	    distance += get_distance(r[-1],initial_location)
	    
	    if distance < shortest_path_distance:
	        shortest_path_distance = distance
	        shortest_path = [initial_location]
	        for i in range(0,len(r)):
	            shortest_path.append(r[i])
	        shortest_path.append(initial_location)

	shortest_path = np.array(shortest_path)
	shortest_path += 1
	shortest_path = list(map(get_city_name, shortest_path))
	        
	print("Shortest path", shortest_path)
	print("Shortest path distance:", shortest_path_distance, "km")
	print("Brute-force execution time:", round(time.time()-start,3), "s")

	return shortest_path_distance

"""
Main program
"""
def main(colony_size, simulations, iterations, evaporation_rate, Q, pr, initial_pheromone, initial_location, verbosity, plot):
	if not colony_size: colony_size = 30
	if not simulations: simulations = 1
	if not iterations: iterations = 100
	if not evaporation_rate: evaporation_rate = 0.5
	if not Q: Q = 20000
	if not verbosity: verbosity = 1
	if not pr: pr = 0.05	
	if not initial_pheromone: initial_pheromone = 1
	if not initial_location: initial_location = 0
	if not plot: plot = 0

	P = generate_matrix(initial_pheromone)

	# best_solution = brute_force(initial_location)
	best_solution = 60858
	
	colony = Colony(colony_size, initial_location, P, evaporation_rate, Q, pr, verbosity, plot)
	
	percentiles = []
	found_iterations = []
	execution_times = []

	for i in range(simulations):
		if verbosity>0:
			print("--------------------------------")
			print("Starting Ant Colony Optimization run", i+1, "of", simulations)
		
		results = colony.run_simulation(iterations, best_solution)	
		
		percentiles.append(results[0])	
		found_iterations.append(results[1])
		execution_times.append(results[2])
		colony.__init__(colony_size, initial_location, P, evaporation_rate, Q, pr, verbosity, plot)

	if verbosity>0:
		print("------------------------------------------")
		print("Simulation runs:", len(percentiles))
		print("Iterations per run:", iterations)
		print("Colony size:", colony_size)
		print("Mean percentile:", np.mean(percentiles))
		print("Mean iteration for best solution:", np.mean(found_iterations))
		print("Mean execution time:", np.mean(execution_times))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Ant Colony Optimization")
	parser.add_argument("-c", type=int, help="Colony size.")
	parser.add_argument("-s", type=int, help="Number of simulation runs.")
	parser.add_argument("-i", type=int, help="Number of iterations per simulation.")
	parser.add_argument("-er", type=float, help="Evaporation rate, how quickly the pheromone evaporates.")
	parser.add_argument("-q", type=int, help="Training parameter Q, amount of pheromones shared by ants.")
	parser.add_argument("-pr", type=int, help="Pheromone deposit to all edges in pheromone update.")
	parser.add_argument("-ip", type=int, help="Initial pheromone level.")
	parser.add_argument("-l", type=int, help="Initial location.")
	parser.add_argument("-v", type=int, help="Verbosity level (0 = no text, 1 = some text, 2 = all text).")
	parser.add_argument("-plt", type=int, help="Add plot (0 = no plot, 1 = plot)")
	args = parser.parse_args()
	main(args.c, args.s, args.i, args.er, args.q, args.pr, args.ip, args.l, args.v, args.plt)