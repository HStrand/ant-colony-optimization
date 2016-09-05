import numpy as np
import pandas as pd
import time

"""
Constants
"""
COLONY_SIZE = 30
ALPHA = 1
BETA = 5
EVAPORATION_RATE = 0.5
Q = 500
INITIAL_PHEROMONE = 1
P_RANDOM_CELL = 0.01

distances = pd.read_csv('distances.csv', sep=';') # Get distances from file
numerated = {}
cities = distances.columns.values
for i in range(1,len(cities)):
    numerated[i] = cities[i]

P = np.array([[INITIAL_PHEROMONE]*len(distances)]*len(distances)) # Initial transition matrix
inverse_identity = 1 - np.identity(8) # Inverse identity matrix
P = P*inverse_identity # Set diagonal to 0 to avoid transitionss to current state

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


class Ant:
	def __init__(self, initial_location):
		self.path = []
		self.initial_location = initial_location
		self.location = initial_location		
		self.distance_travelled = 0
		self.matrix = P 
		""" 
		The ant keeps a copy of the transition matrix. 
		This is also used to keep track of the places the ant has visited. 
		THe initial location has to be the last stop.
		"""
		self.matrix[:,self.location]=0 # Prevent the ant from moving back to the initial location until the last step.

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
		print("Travelling", distance, "km from", numerated[self.location+1], "to", numerated[self.initial_location+1])
		self.path.append(self.initial_location)
		self.distance_travelled += distance
		self.location = self.initial_location
		print("Path:", self.path)
		print("Distance travelled:", self.distance_travelled, "km")


if __name__ == '__main__':
	start = time.time()
	ant = Ant(0)
	ant.march()
	print("Time:", time.time()-start)