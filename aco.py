import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

"""
Training parameters
"""
COLONY_SIZE = 10
# ALPHA = 1
# BETA = 5
EVAPORATION_RATE = 0.5
Q = 20000
INITIAL_PHEROMONE = 1
P_RANDOM_CELL = 0.05
ITERATIONS = 500

distances = pd.read_csv('distances.csv', sep=';') # Get distances from file
numerated = {}
cities = distances.columns.values
for i in range(1,len(cities)):
    numerated[i] = cities[i]

INVERSE_IDENTITY = 1 - np.identity(len(distances)) # Inverse identity matrix

def generate_matrix():
	P = np.array([[INITIAL_PHEROMONE]*len(distances)]*len(distances)) # Initial transition matrix	
	P = P * INVERSE_IDENTITY # Set diagonal to 0 to avoid transitionss to current state
	return P

P = generate_matrix()

"""
Helper functions
"""
def get_row(city):
	return np.array(distances)[numerated[city],1:]

def get_row_by_num(num):
	return np.array(distances)[num,1:]

def get_column(city):
	return np.array(distances[city])

def get_distance(a, b):
	return get_row_by_num(a)[b]

def get_city_name(x):
    return numerated[x]


class Ant:
	def __init__(self, initial_location):
		self.path = [initial_location] # Path starts with initial location.
		self.initial_location = initial_location
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
		for step in range(0,len(P)-1): # -1 since initial location is not available.
			self.step()

		# Finally, move back to initial location.
		distance = get_distance(self.location, self.initial_location)
		if VERBOSE:
			print("Travelling", distance, "km from", numerated[self.location+1], "to", numerated[self.initial_location+1])
		self.path.append(self.initial_location)
		self.distance_travelled += distance
		self.location = self.initial_location

	def get_grid(self):
		self.matrix = np.empty_like(P)
		self.matrix[:] = P
		self.matrix[:,self.location]=0 # Prevent the ant from moving back to the initial location until the last step.


class Colony:
	def __init__(self, colony_size, initial_location):
		self.ants = []
		for i in range(colony_size):
			self.ants.append(Ant(initial_location))		
		self.paths = []
		self.path_distances = []
		self.shortest_path = []
		self.shortest_path_distance = 999999999
		self.global_shortest_path = []
		self.global_shortest_path_distance = 999999999
		self.global_shortest_iteration = 0
		self.initial_location = initial_location
		self.iteration = 1

	def march(self):
		# Reset stored iteration values.
		if(self.iteration>0):
			self.paths = []
			self.path_distances = []
			self.shortest_path = []
			self.shortest_path_distance = 999999999
			for ant in self.ants:
				ant.__init__(self.initial_location) # Ants have no memory.

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

	def update_pheromone_trail(self, grid):
		grid = evaporate(grid)
		grid = self.deposit_pheromones(grid)

		return grid

	def deposit_pheromones(self, grid):
		for ant in self.get_top_ants(5):
			for i in range(0,len(ant.path)-1):
				grid[ant.path[i],ant.path[i+1]] += Q/ant.distance_travelled # Update rule

		grid += P_RANDOM_CELL # All edges receive a small amount of pheromones to ensure some randomness
		grid = grid * INVERSE_IDENTITY # Remove self-loops to nodes

		return grid

	def get_top_ants(self, count):
		sorted_ants = sorted(colony.ants, key=lambda x: x.distance_travelled, reverse=False)
		return sorted_ants[:count]


def evaporate(grid):
	return grid*(1-EVAPORATION_RATE)



if __name__ == '__main__':
	VERBOSE = False
	start = time.time()
	
	colony = Colony(COLONY_SIZE, 0)
	global_shortest_distances = []
	avg_distances = []
	iterations = []

	for i in range(ITERATIONS):
		print("Starting iteration", colony.iteration)
		colony.march()
		avg = colony.score()
		avg_distances.append(avg)
		global_shortest_distances.append(colony.global_shortest_path_distance)
		iterations.append(i)
		P = colony.update_pheromone_trail(P)

	best_path = np.array(colony.global_shortest_path)
	best_path += 1
	best_path = list(map(get_city_name, best_path))

	print("Shortest path found:", best_path)
	print("Found in iteration", colony.global_shortest_iteration)
	print("Shortest path distance:", colony.global_shortest_path_distance, "km")

	print("Time:", time.time()-start)

	iterations = np.array(iterations)
	global_shortest_distances = np.array(global_shortest_distances)
	plt.plot(iterations, global_shortest_distances)
	plt.plot(iterations, avg_distances)
	plt.show()