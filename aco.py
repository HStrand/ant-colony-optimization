import numpy as np
import time

MATRIX = np.array([[0,0.4,0.3,0.2,0.1], [0.1,0,0.4,0.3,0.2], [0.1,0.2,0,0.3,0.4], [0.1,0.3,0.2,0,0.4], [0.3,0.4,0.2,0.1,0]])

class Ant:
	def __init__(self, trail):
		self.path = []
		self.location = 0
		self.matrix = MATRIX 
		""" 
		The ant keeps a copy of the transition matrix. 
		This is also used to keep track of the places the ant has visited. 
		"""

	def step(self):
		transition_probabilities = self.matrix[self.location]

		step = -1 # Intentionally crash the program if something is wrong
		
		# Weighted random sampling:
		goal = np.random.uniform()*sum(transition_probabilities)
		cumulative = 0
		for i in range(0,len(transition_probabilities)):
			cumulative += transition_probabilities[i]
			if(cumulative>goal):
				step = i
				break
		
		print("Step to", step)
		self.location = step # Move ant to the new location.
		self.matrix[:,step] = 0 # Ensure ant will not visit this place again.

	def walk(self):
		for step in range(0,len(MATRIX)):
			self.step()


if __name__ == '__main__':
	start = time.time()
	ant = Ant(None)
	ant.walk()
	#for i in range(0,len(MATRIX)):
	#	ant.step()
	print(time.time()-start)