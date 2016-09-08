import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import time
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
def brute_force():
	cities = []
	initial_location = 4
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
def main(colony_size, iterations, evaporation_rate, Q, pr, initial_pheromone, verbosity):
	if not colony_size: colony_size = 30
	if not iterations: iterations = 100
	if not evaporation_rate: evaporation_rate = 0.5
	if not Q: Q = 20000
	if not verbosity: verbosity = 1
	if not pr: pr = 0.05	
	if not initial_pheromone: initial_pheromone = 1

	P = generate_matrix(initial_pheromone)

	# best_solution = brute_force()
	best_solution = 60858

	if(verbosity>0):
		print("--------------------------------")
		print("Starting Ant Colony Optimization")
		print("Colony size:", colony_size)
		print("Number of iterations:", iterations)
	start = time.time()
	
	colony = Colony(colony_size, 4, P, evaporation_rate, Q, pr, verbosity)
	results = colony.run_simulation(iterations)

	global_shortest_distances = results[0]
	avg_distances = results[1]
	iteration_list = results[2]

	best_path = np.array(colony.global_shortest_path) + 1
	best_path = list(map(get_city_name, best_path))

	if(verbosity>0):
		print("Shortest path found:", best_path)
		print("Found in iteration", colony.global_shortest_iteration)
		print("Shortest path distance:", colony.global_shortest_path_distance, "km")
		print("Percentile:", round(best_solution*100/colony.global_shortest_path_distance,2))
		print("Ant Colony Optimization execution time:", round(time.time()-start,3), "s")

	iteration_list = np.array(iteration_list)
	global_shortest_distances = np.array(global_shortest_distances)
	plt.plot(iteration_list, global_shortest_distances)
	plt.plot(iteration_list, avg_distances)
	plt.show()


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Ant Colony Optimization")
	parser.add_argument("-c", type=int, help="Colony size.")
	parser.add_argument("-i", type=int, help="Number of iterations.")
	parser.add_argument("-er", type=float, help="Evaporation rate, how quickly the pheromone evaporates.")
	parser.add_argument("-q", type=int, help="Training parameter Q, amount of pheromones shared by ants.")
	parser.add_argument("-pr", type=int, help="Pheromone deposit to all edges in pheromone update.")
	parser.add_argument("-ip", type=int, help="Initial pheromone level.")
	parser.add_argument("-v", type=int, help="Verbosity level (0 = no text, 1 = some text, 2 = all text).")	
	args = parser.parse_args()
	main(args.c, args.i, args.er, args.q, args.pr, args.ip, args.v)