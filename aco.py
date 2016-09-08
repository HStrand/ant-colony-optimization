import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import time


distances = pd.read_csv('distances2.csv', sep=';') # Get distances from file
numerated = {}
cities = distances.columns.values
for i in range(1,len(cities)):
    numerated[i] = cities[i]

INVERSE_IDENTITY = 1 - np.identity(len(distances)) # Inverse identity matrix


"""
Ant class
"""
class Ant:
	def __init__(self, initial_location, colony):
		self.path = [initial_location] # Path starts with initial location.
		self.initial_location = initial_location
		self.colony = colony
		self.location = initial_location		
		self.distance_travelled = 0		
		self.get_grid()
		""" 
		The ant gets a copy of the transition matrix. 
		This is also used to keep track of the places the ant has visited. 
		THe initial location has to be the last stop.
		"""		

	def step(self):
		transition_probabilities = self.matrix[self.location]

		step = -1 # Intentionally crash the program if something is wrong.
		
		# Weighted random sampling:
		goal = np.random.uniform()*sum(transition_probabilities)
		cumulative = 0
		for i in range(0,len(transition_probabilities)):
			cumulative += transition_probabilities[i]
			if(cumulative>goal):
				step = i
				break
		
		distance = get_distance(self.location, step) # Record distance travelled.
		if VERBOSE:
			print("Travelling", distance, "km from", numerated[self.location+1], "to", numerated[step+1])

		self.location = step # Move ant to the new location.
		self.matrix[:,step] = 0 # Ensure ant will not visit this place again.
		self.path.append(step)
		self.distance_travelled += distance

	def march(self):
		for step in range(0,len(self.matrix)-1): # -1 since initial location is not available.
			self.step()

		# Finally, move back to initial location.
		distance = get_distance(self.location, self.initial_location)
		if VERBOSE:
			print("Travelling", distance, "km from", numerated[self.location+1], "to", numerated[self.initial_location+1])
		self.path.append(self.initial_location)
		self.distance_travelled += distance
		self.location = self.initial_location

	def get_grid(self):
		self.matrix = np.empty_like(self.colony.grid)
		self.matrix[:] = self.colony.grid
		self.matrix[:,self.location]=0 # Prevent the ant from moving back to the initial location until the last step.

"""
Ant Colony class
"""
class Colony:
	def __init__(self, colony_size, initial_location, grid, Q):
		self.colony_size = colony_size
		self.grid = grid
		self.Q = Q
		self.paths = []
		self.path_distances = []
		self.shortest_path = []
		self.shortest_path_distance = 999999999
		self.global_shortest_path = []
		self.global_shortest_path_distance = 999999999
		self.global_shortest_iteration = 0
		self.initial_location = initial_location
		self.iteration = 1
		self.ants = []
		for i in range(colony_size):
			self.ants.append(Ant(initial_location, self))	

	def march(self):
		# Reset stored iteration values.
		if(self.iteration>0):
			self.paths = []
			self.path_distances = []
			self.shortest_path = []
			self.shortest_path_distance = 999999999
			for ant in self.ants:
				ant.__init__(self.initial_location, self) # Ants have no memory.

		# March the ants.
		for ant in self.ants:
			ant.march()
			if VERBOSE:
				print("Distance:", ant.distance_travelled, "km")
			self.paths.append(ant.path)
			self.path_distances.append(ant.distance_travelled)
			
			# Check if path is iteration best.
			if(ant.distance_travelled<self.shortest_path_distance):
				self.shortest_path_distance = ant.distance_travelled
				self.shortest_path = ant.path
				# Check if path is global best.
				if(ant.distance_travelled<self.global_shortest_path_distance):
					self.global_shortest_path_distance = ant.distance_travelled
					self.global_shortest_path = ant.path
					self.global_shortest_iteration = self.iteration # Record when global best was hit

		self.iteration += 1

	def score(self):
		if VERBOSE:
			print("-----------------------")
			print("Shortest path:", self.shortest_path)
			print("Shortest distance:", self.shortest_path_distance, "km")
			print("Average distance:", round(np.mean(self.path_distances),2), "km")

		return round(np.mean(self.path_distances),2)

	def update_pheromone_trail(self):
		self.grid = self.grid*(1-EVAPORATION_RATE)
		self.deposit_pheromones()

	def deposit_pheromones(self):
		for ant in self.get_top_ants(int(self.colony_size/2)):
			for i in range(0,len(ant.path)-1):
				self.grid[ant.path[i],ant.path[i+1]] += self.Q/ant.distance_travelled # Update rule

		self.grid += P_RANDOM_CELL # All edges receive a small amount of pheromones to ensure some randomness
		self.grid = self.grid * INVERSE_IDENTITY # Remove self-loops to nodes

	def get_top_ants(self, count):
		sorted_ants = sorted(self.ants, key=lambda x: x.distance_travelled, reverse=False)
		return sorted_ants[:count]

	def run_simulation(self, iterations):
		global_shortest_distances = []
		avg_distances = []
		iteration_list = []

		for i in range(iterations):
			if(self.iteration*100/iterations%10==0):
				print(round(self.iteration*100/iterations,0), "%")
			self.march()
			avg = self.score()
			avg_distances.append(avg)
			global_shortest_distances.append(self.global_shortest_path_distance)
			iteration_list.append(i)
			self.update_pheromone_trail()

		return global_shortest_distances, avg_distances, iteration_list

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
Main program
"""
def main(colony_size, iterations, evaporation_rate, Q, pr, initial_pheromone, verbosity):
	if not colony_size: colony_size = 30
	if not iterations: iterations = 100
	if not evaporation_rate: evaporation_rate = 0.5
	if not Q: Q = 20000
	if not verbosity: verbosity = 0
	if not pr: pr = 0.05	
	if not initial_pheromone: initial_pheromone = 1
	
	global VERBOSE
	if(verbosity==0):
		VERBOSE = False
	elif(verbosity==1):
		VERBOSE = True

	global P_RANDOM_CELL
	P_RANDOM_CELL = pr

	global EVAPORATION_RATE
	EVAPORATION_RATE = evaporation_rate

	P = generate_matrix(initial_pheromone)

	# best_solution = brute_force()
	best_solution = 46508

	print("--------------------------------")
	print("Starting Ant Colony Optimization")
	print("Number of iterations:", iterations)
	start = time.time()
	
	colony = Colony(colony_size, 4, P, q)
	results = colony.run_simulation(iterations)

	global_shortest_distances = results[0]
	avg_distances = results[1]
	iteration_list = results[2]

	best_path = np.array(colony.global_shortest_path) + 1
	best_path = list(map(get_city_name, best_path))

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