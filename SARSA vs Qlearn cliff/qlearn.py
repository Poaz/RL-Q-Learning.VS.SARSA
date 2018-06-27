import random

#Reference for the core code: https://github.com/studywolf/blog/tree/master/RL/SARSA%20vs%20Qlearn%20cliff
#Code is modified for our needs.

#The Q-Learning class
class QLearn:
    #Constructor for the Q-Learning class
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.Qdata = [] #own implementation


    #Getter for the Q-Value.
    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    #Function that set epsilon to 0, meaning that it won't take random actions. Made by us
    def stopExporation(self):
        self.epsilon = 0

    #Function that calculates the Q-value. This is the Q-Learning function.
    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    #Function that takes an actions based on the Q-value.
    def chooseAction(self, state):
        #But everytime there is a chance that it will take a random action
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            #Get the Q-value for the available actions
            q = [self.getQ(state, a) for a in self.actions]
            #Picks the highest value.
            maxQ = max(q)
            #If there are two there shares the highest value, one will be picked.
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            #Stores data for the plots. own implementation
            self.Qdata.append(i)
            action = self.actions[i]
        return action

    #Updates the Q-value for the current state and actions, according to the next state.
    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    #Getter for the QData, used for making the plots. own implementation
    def getQData(self):
        tmp = float(sum(self.Qdata)) / max(len(self.Qdata), 1)
        self.Qdata = []
        return tmp
