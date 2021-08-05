# Example 4.2: Jack's Car Rental   Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars. If Jack has a car available, he rents it out and is credited $10 by the national company. If he is out of cars at that location, then the business is lost. Cars become available for renting the day after they are returned. To help ensure that cars are available where they are needed, Jack can move them between the two locations overnight, at a cost of $2 per car moved. We assume that the number of cars requested and returned at each location are Poisson random variables, meaning that the probability that the number is  is , where $\lambda $ is the expected number. Suppose $\lambda $ is 3 and 4 for rental requests at the first and second locations and 3 and 2 for returns. To simplify the problem slightly, we assume that there can be no more than 20 cars at each location (any additional cars are returned to the nationwide company, and thus disappear from the problem) and a maximum of five cars can be moved from one location to the other in one night. We take the discount rate to be  and formulate this as a continuing finite MDP, where the time steps are days, the state is the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved between the two locations overnight. Figure  4.4 shows the sequence of policies found by policy iteration starting from the policy that never moves any cars.

from random import seed, random
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp
from itertools import product, accumulate
from copy import deepcopy
from scipy.stats import poisson

np.set_printoptions(threshold=1000, linewidth=200, precision=2, suppress=True)

def pois(n, y, cache={}):
	if y not in cache:
		cache[y] = [( y**n_/factorial(n_) )*exp(-y) for n_ in range(20)]
		cache[y] += [1-sum(cache[y])]
	return cache[y][n]


class Agent:
	def __init__(self):
		self.Exp_R = [[]] # Rewards
		self.P_T = [[]] # Transition Probabilities
		self.Exp_V = np.zeros((21, 21))
		self.R_MOVE = -2
		R_RENT = 10 
		
		## Create Expected reward[total_cars_i][total_cars_j] -> float
		P_sale_i = poisson.pmf(range(21), mu=3).reshape(21, 1)
		P_sale_j = poisson.pmf(range(21), mu=4).reshape(1, 21)
		P_sale = np.ones((21, 21))*P_sale_i*P_sale_j

		cars_sold = np.arange(21).reshape((21, 1)) \
					+ np.arange(21).reshape((1, 21))
		R_sale = R_RENT * cars_sold
		
		# Selling 5 cars means you can also sell 4, 3, ...
		self.Exp_R = np.cumsum(np.cumsum(P_sale*R_sale, axis=0), axis=1)

		# If we have 4 cars, but 5 customers, we still sell 4 cars, Therefore:
		# "np.cumsum(P_sale_i)" yields the sales already accounted for. 
		# "1-np.cumsum(P_sale_i)" therefore yields unaccounted sales, i.e. tail probabilities
		P_tails_i = (1-np.cumsum(P_sale_i)).reshape(21, 1)
		P_tails_j = (1-np.cumsum(P_sale_j)).reshape(1, 21)
		# Distribute P_i over the length of j, and vice versa, to get expected reward
		P_tails = P_tails_i @ P_sale_j + P_sale_i @ P_tails_j
		# Multiply by reward for selling the max available cars
		self.Exp_R += R_sale*P_tails

		## Create Transition Probabilities Cars[cars_i, cars_j] -> Probability Array[end_i, end_j]

		## 1.) Consider car reduction due to sales
		# Reverse P_sale, since ending up with 0 cars requires selling all cars, i.e. last idx
		# Pad zeros because selling doesn't add cars / can't surpass cars_i
		P_T_sale_i = np.array([P_sale_i.flatten()[i::-1].tolist() + [0]*(20-i) for i in range(21)])
		# Add P(sale > cars_i) to P(end_i = 0)
		P_T_sale_i[:,0] += P_tails_i.flatten()
		# P_T_i[1] -> [0.95, 0.05, 0, ...]
		# i.e. if cars_i == 0, then P(end_i=0) = 95% and P(end_i=1) = 5%

		## 2.) Consider car increase due to returns
		# If we are at cars_i, then P(cars_i-1) = 0 and P(cars_i+0) ~ Pois.pmf(x=0, mu=3)
		
		P_T_return_i = np.array([[0]*i + poisson.pmf(range(21-i), mu=3).tolist() for i in range(21)])
		# Add P(returns + cars_i > 20) to P(cars_i = 20)
		P_T_return_i[:,-1] += 1-poisson.cdf(range(20, -1, -1), mu=3)

		# To create transition T(cars_i=1), we need to add the 95% from end_i=0 5% of end_i=1
		# I.e. we need to sum 0.95*P_T_return_i[0] + 0.05*P_T_return_i[1]
		P_T_i = np.array([(P_T_sale_i[car_i]*P_T_return_i.T).sum(axis=1) for car_i in range(21)])

		# Do for j as we did for i
		P_T_sale_j = np.array([P_sale_j.flatten()[j::-1].tolist() + [0]*(20-j) for j in range(21)])
		P_T_sale_j[:,0] += P_tails_j.flatten()

		P_T_return_j = np.array([[0]*j + poisson.pmf(range(21-j), mu=3).tolist() for j in range(21)])
		P_T_return_j[:,-1] += 1-poisson.cdf(range(20, -1, -1), mu=2)
		P_T_j = np.array([(P_T_sale_j[car_j]*P_T_return_j.T).sum(axis=1) for car_j in range(21)])

		# Combine so we can index P_T[cars_i, cars_j]
		self.P_T = np.array([[P_T_i[i:i+1].T @ P_T_j[j:j+1] for j in range(21)] for i in range(21)])

		
		


	def R(self, cars_i, cars_j, move_i, move_j):
		cars_i += move_j - move_i
		cars_j += move_i - move_j
		return self.R_MOVE*(move_i+move_j) + self.Exp_R[cars_i, cars_j]
		
	def V(self, cars_i, cars_j, move_i, move_j):
		cars_i += move_j - move_i
		cars_j += move_i - move_j
		return (self.P_T[cars_i, cars_j] * self.Exp_V).sum()
		
	def play(self):
		n = 1
		V2 = np.zeros((21, 21))
		policy = np.zeros((21, 21))
		y = 0.9
		# V(s) = E[R + y*V(s+1)]

		for epoch in range(1, n+1):
			for cars_i, cars_j in product(range(21), range(21)):
				moves_i = range(1+min(20-cars_j, min(5, cars_i)))
				moves_j = range(1+min(20-cars_i, min(5, cars_j)))
				state_actions = [(cars_i, cars_j, move_i, move_j) \
								for move_i, move_j in product(moves_i, moves_j) \
								if move_i == 0 or move_j == 0]
				val, a = max([(
							self.R(*sa) + y*self.V(*sa), 
							sa[-2:]
							) for sa in state_actions])
				policy[cars_i, cars_j] = a[0]-a[1]
				V2[cars_i, cars_j] = val
			self.Exp_V += (1/epoch)*(V2-self.Exp_V)
		X, Y, C = zip(*[[i, j, policy[i, j]] for i, j in product(range(21), range(21))])
		plt.scatter(X, Y, c=C, cmap='Greys')
		plt.gcf().set_size_inches(2, 2)
		plt.title('Cars to move. White means move from i, black means move from j')
		#plt.show()
		#print(np.array([row[::-1] for row in policy]))

A = Agent()
A.play()
