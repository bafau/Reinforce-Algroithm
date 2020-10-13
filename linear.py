import gym
import numpy as np
import matplotlib.pyplot as plt

def linear(seed):
	NUM_RUNS = 2000
	LEARNING_RATE = 0.002
	GAMMA = 0.99
	
	
	# Create gym and seed numpy
	#env = gym.make('CartPole-v0')
	env = gym.make('LunarLander-v2')
	
	np.random.seed(seed)
	
	#data of the environment
	nA = env.action_space.n
	nS = env.observation_space.shape[0]
	
	# Init weight
	theta = np.random.rand(nA, nS)
	
	
	def softmax(state,theta):
		# Linear Combination of Features
		z = theta.dot(state.T)
		
		# shift z by maximum to improve numeric stability 
		#(https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
		shiftz = z - np.max(z)
		
		exp = np.exp(shiftz)
		return exp/np.sum(exp)
		
	def gradient_softmax(probs):
		# create diagonal matrix with probability Values in the diagonal 
		# and substract the dot product
		return np.diagflat(probs) - np.dot(probs, probs.T) #[[p1,0];[0,p2]] - [[p1p1, p1p2];[p2p1, p2,p2]]
		
	#total rewards; stores the total reward  of each episode
	total_rewards = []

	for j in range(NUM_RUNS):
		
		#Initialize Environment
		state = env.reset()[None,:]
		
		#Store gradient and reward of the episode
		grads = []	
		rewards = []
		
		# stores the reward of the episode
		score = 0
		
		#Run interaction until Episode is finished
		
		while True:


			# Sample from policy and take action in environment
			probs = softmax(state,theta)
			action = np.random.choice(nA,p=probs.T[0])

			next_state,reward,done,_ = env.step(action)
			next_state = next_state[None,:]

			# Compute gradient 
			#choos the row of the action
			dsoftmax = gradient_softmax(probs)[action, :]
		
			
			#Compute Gradietn (chapter ...)
			gradient  = dsoftmax.reshape(nA,1).dot(state) # nA x nS matrix
			
			# log gradient for the REINFORCE Update
			grad = gradient / probs[action,0] 
			

			grads.append(grad)
			rewards.append(reward)		

			score+=reward

			# update state
			state = next_state

			if done:
				break
				
		# REINFORCE Update
		for i in range(len(grads)):

			 theta += LEARNING_RATE * grads[i] * sum([ r * (GAMMA ** t) for t,r in enumerate(rewards[i:])])
		
		total_rewards.append(score)
		 
		print('episode: ', j,'score: %.1f' % score)

		env.close()
		
	return total_rewards
		



