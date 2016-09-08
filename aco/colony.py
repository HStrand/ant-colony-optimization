import time
from .ant import Ant
from main import *

"""
Ant Colony class
"""
class Colony:
	def __init__(self, colony_size, initial_location, grid, evaporation_rate, Q, pr, verbosity, plot):
		self.colony_size = colony_size
		self.grid = grid
		self.evaporation_rate = evaporation_rate
		self.Q = Q
		self.pr = pr
		self.verbosity = verbosity
		self.plot = plot
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
			if self.verbosity>2:
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
		if self.verbosity>1:
			print("-----------------------")
			print("Shortest path:", self.shortest_path)
			print("Shortest distance:", self.shortest_path_distance, "km")
			print("Average distance:", round(np.mean(self.path_distances),2), "km")

		return round(np.mean(self.path_distances),2)

	def update_pheromone_trail(self):
		self.grid = self.grid*(1-self.evaporation_rate)
		self.deposit_pheromones()

	def deposit_pheromones(self):
		for ant in self.get_top_ants(int(self.colony_size/2)):
			for i in range(0,len(ant.path)-1):
				self.grid[ant.path[i],ant.path[i+1]] += self.Q/ant.distance_travelled # Update rule

		self.grid += self.pr # All edges receive a small amount of pheromones to ensure some randomness
		self.grid = self.grid * INVERSE_IDENTITY # Remove self-loops to nodes

	def get_top_ants(self, count):
		sorted_ants = sorted(self.ants, key=lambda x: x.distance_travelled, reverse=False)
		return sorted_ants[:count]

	def simulate(self, iterations):
		global_shortest_distances = []
		avg_distances = []
		iteration_list = []

		for i in range(iterations):			
			if(self.iteration*100/iterations%10==0 and self.verbosity>1):
				print(round(self.iteration*100/iterations,0), "%")
			self.march()
			avg = self.score()
			avg_distances.append(avg)
			global_shortest_distances.append(self.global_shortest_path_distance)
			iteration_list.append(i)
			self.update_pheromone_trail()

		return global_shortest_distances, avg_distances, iteration_list

	def run_simulation(self, iterations, best_solution):
		start = time.time()

		if(self.verbosity>0):
			print("Colony size:", self.colony_size)
			print("Number of iterations:", iterations)	

		results = self.simulate(iterations)

		global_shortest_distances = results[0]
		avg_distances = results[1]
		iteration_list = results[2]

		best_path = np.array(self.global_shortest_path) + 1
		best_path = list(map(get_city_name, best_path))

		percentile = round(best_solution*100/self.global_shortest_path_distance,2)
		execution_time = round(time.time()-start,3)

		if(self.verbosity>0):
			print("Shortest path found:", best_path)
			print("Found in iteration", self.global_shortest_iteration)
			print("Shortest path distance:", self.global_shortest_path_distance, "km")
			print("Percentile:", percentile)
			print("Ant Colony Optimization execution time:", execution_time, "s")

		if(self.plot):
			iteration_list = np.array(iteration_list)
			global_shortest_distances = np.array(global_shortest_distances)
			plt.plot(iteration_list, global_shortest_distances)
			plt.plot(iteration_list, avg_distances)
			plt.show()

		return percentile, self.global_shortest_iteration, execution_time