#Imports
import random

#Declaration
#Reference for the core code: https://github.com/studywolf/blog/tree/master/RL/SARSA%20vs%20Qlearn%20cliff
#Code is modified for our needs.

#Class Sarsa
class SARSA:

    #Initializing function to set the parameters of the algorithm.
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.sarsadata = [] #own implementation

    #Get q value function to return the q value
    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    #Function for stopping exploration setting epsilon to zero. own implementation.
    def stopExporation(self):
        self.epsilon = 0

    #Learn the Q, this is the SARSA algorithm.
    def learnQ(self, state, action, reward, value):
        oldValue = self.q.get((state, action), None)
        if oldValue is None:
            self.q[(state, action)] = reward 
        else: #calculates the Q Value.
            self.q[(state, action)] = oldValue + self.alpha * (value - oldValue)

    #Function for choosing the action
    def chooseAction(self, state):
        #Exploration - Do a random action if the random number is ssmaller than epsilon
        if random.random() < self.epsilon:
            action = random.choice(self.actions) #takes a random action
        else:
            #calculates the available actions Q values.
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)

            #If there is more than one action, it will pick a random action.
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            #Capture data for plots. own implementation
            self.sarsadata.append(i)
            action = self.actions[i]
        return action

    #Learning function gets the next Q-value for the next action. To look ahead.
    def learn(self, state_1, action_1, reward, state_2, action_2):
        nextQ = self.getQ(state_2, action_2)
        self.learnQ(state_1, action_1, reward, reward + self.gamma * nextQ)

    #Function to get the Data for plots. own implementation
    def getSaesaData(self):
        tmp = float(sum(self.sarsadata)) / max(len(self.sarsadata), 1)
        self.sarsadata = []
        return tmp
