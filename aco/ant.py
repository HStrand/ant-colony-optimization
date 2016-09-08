import numpy as np
from main import *

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
		if self.colony.verbosity>2:
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
		if self.colony.verbosity>2:
			print("Travelling", distance, "km from", numerated[self.location+1], "to", numerated[self.initial_location+1])
		self.path.append(self.initial_location)
		self.distance_travelled += distance
		self.location = self.initial_location

	def get_grid(self):
		self.matrix = np.empty_like(self.colony.grid)
		self.matrix[:] = self.colony.grid
		self.matrix[:,self.location]=0 # Prevent the ant from moving back to the initial location until the last step.