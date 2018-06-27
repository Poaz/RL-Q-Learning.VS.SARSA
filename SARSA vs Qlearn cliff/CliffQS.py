import cellular
import qlearn
import sarsa
import time
import sys
import Plotter
import thread
import matplotlib.pyplot as plt
import numpy as np

startCell = None

'''
class representing each cell in the gridworld inherits from cellular.Cell
'''
class Cell(cellular.Cell):
    def __init__(self):
        self.cliff = False
        self.goal = False
        self.wall = False
        self.swamp = False  #Own Implementation

    #paint the cells in pretty colours
    def colour(self):
        if self.cliff:
            return 'red'
        if self.goal:
            return 'green'
        if self.wall:
            return 'black'
        if self.swamp:          #Own Implementation
            return 'brown'      #Own Implementation
        else:
            return 'white'

    #translate chracters into map booleans
    def load(self, data):
        global startCell
        if data == 'S':
            startCell = self
        if data == '.':
            self.wall = True
        if data == 'X':
            self.cliff = True
        if data == 'w':         #Own Implementation
            self.swamp = True   #Own Implementation
        if data == 'G':
            self.goal = True


'''
this agent has a Q-Learning Ai, it inherits from cellular.Agent
'''
class QAgent(cellular.Agent):

    #constructor
    def __init__(self):
        self.ai = qlearn.QLearn(
            actions=range(directions), epsilon=0.1, alpha=0.1, gamma=0.9)
        self.lastAction = None
        self.score = 0
        self.episode = 0                #Own Implementation
        self.QAverageQ = []             #Own Implementation
        self.actionsInEpisode = 0       #Own Implementation
        self.actionsInAllEpisodes = []  #Own Implementation

    def colour(self):
        return 'purple'


    '''
    function for updating each timestep
    '''
    def update(self):
        reward = self.calcReward()
        state = self.calcState()
        action = self.ai.chooseAction(state) #returns the action we want to tage.
        if self.lastAction is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, state) #if we have taken an action learn from it.
        self.lastState = state
        self.lastAction = action
        self.actionsInEpisode += 1  #Own Implementation
        here = self.cell
        if here.goal or here.cliff: #termination state
            self.episode += 1       #Own Implementation
            self.cell = startCell
            self.QAverageQ.append(self.ai.getQData())       #Own Implementation
            self.actionsInAllEpisodes.append(self.actionsInEpisode) #Own Implementation
            self.actionsInEpisode = 0   #Own Implementation
            self.lastAction = None
        else:
            self.goInDirection(action) #execute action

    def calcState(self):
        return self.cell.x, self.cell.y


    #returns the rewards for each cell type.
    def calcReward(self):
        here = self.cell
        if here.cliff:
            return cliffReward
        elif here.goal:
            self.score += 1
            return goalReward
        elif here.swamp:        #Own Implementation
            return swampReward  #Own Implementation
        else:
            return normalReward


'''
this class is another agent class that uses sarsa learning instead of Q-learning
'''
class SAgent(cellular.Agent):

    #constructor
    def __init__(self):
        self.ai = sarsa.SARSA(
            actions=range(directions), epsilon=0.1, alpha=0.1, gamma=0.9)
        self.lastAction = None
        self.score = 0
        self.episode = 0                #Own Implementation
        self.SarsaAverageQ = []         #Own Implementation
        self.actionsInEpisode = 0       #Own Implementation
        self.actionsInAllEpisodes = []  #Own Implementation


    def colour(self):
        return 'blue'


    '''
    function for updating each timestep
    '''
    def update(self):
        reward = self.calcReward()
        state = self.calcState()
        action = self.ai.chooseAction(state) #returns the action we want to tage.
        if self.lastAction is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, state, action) #if we have taken an action in last timestep learn from it.
        self.lastState = state
        self.lastAction = action
        self.actionsInEpisode += 1  #Own Implementation
        here = self.cell
        if here.goal or here.cliff:   #termination state
            self.episode += 1       #Own Implementation
            self.cell = startCell
            self.SarsaAverageQ.append(self.ai.getSaesaData())       #Own Implementation
            self.actionsInAllEpisodes.append(self.actionsInEpisode) #Own Implementation
            self.actionsInEpisode = 0                               #Own Implementation
            self.lastAction = None
        else:
            self.goInDirection(action) #execute action

    def calcState(self):
        return self.cell.x, self.cell.y

    #returns the rewards for each cell type.
    def calcReward(self):
        here = self.cell
        if here.cliff:
            return cliffReward
        elif here.goal:
            self.score += 1
            return goalReward
        elif here.swamp:            #Own Implementation
            return swampReward      #Own Implementation
        else:
            return normalReward



#setup rewards:
normalReward = -1
swampReward = -2
cliffReward = -10
goalReward = 0

'''
Setup world and attach agents to the world.
'''
directions = 4
world = cellular.World(Cell, directions=directions, filename='cliff.txt')
if startCell is None:
    print "You must indicate where the agent starts by putting a 'S' in the map file"
    sys.exit() #the game cant start without a start position, so exit if there is none
QAgent = QAgent()
SAgent = SAgent()
world.addAgent(QAgent, cell=startCell)
world.addAgent(SAgent, cell=startCell)


'''
start training
traning is not done in episodes but in timesteps.
'''
pretraining = 500000
for i in range(pretraining):
    if i % 1000 == 0:
        procent = (i * 100) / pretraining
        sys.stdout.write("%s\r" + str(procent) + "% done training")
        sys.stdout.flush()

        if (QAgent.score > 10): #if the Ai is good enough stop exploring    #Own Implementation
            QAgent.ai.stopExporation()                                      #Own Implementation
        if (SAgent.score > 10):                                             #Own Implementation
            SAgent.ai.stopExporation()                                      #Own Implementation
    world.update()  #next timestep
#end of traning


#plot traning data

#Next section is all our own Implementation
plotContainer = Plotter.PlotContainer(2)
plt0 = Plotter.Plot( "Averages for Q-Learning", QAgent.QAverageQ, 'Episodes', 'Average Q value for the episode')
plotContainer.plots.append(plt0)

plt1 = Plotter.Plot( "Actions per episode for Q-Learning", QAgent.actionsInAllEpisodes, 'Episodes', 'Actions')
plotContainer.plots.append(plt1)

plt2 = Plotter.Plot( "Averages for SARSA", SAgent.SarsaAverageQ, 'Episodes', 'Average Q value for the episode')
plotContainer.plots.append(plt2)

plt3 = Plotter.Plot( "Actions per episode for SARSA", SAgent.actionsInAllEpisodes, 'Episodes', 'Actions')
plotContainer.plots.append(plt3)


plotContainer.start() #start plot thread contaning all the plots.


#Launch game time to watch.
world.display.activate(size=30)
world.display.delay = 1

while 1:
    world.update()