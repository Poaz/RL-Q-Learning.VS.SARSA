'''
This entire file is our own implementation
It uses multithreading to show the plots without locking the execution of the main game.
'''
import threading
import matplotlib.pyplot as plt
import numpy as np



class PlotContainer(threading.Thread):

    def __init__(self, threadID=None):
        threading.Thread.__init__(self)  #initialize the thread
        self.threadID = threadID
        self.plots = []

    def PlotAllPlotsInOneWindow(self):
        for i in range(0,len(self.plots),1):  #foreach plot :
            plt.subplot(221 + i)        #221 means that the subplot should be part of a 2 by 2 grid of subplots the 1 means its the first.
            plt.plot(self.plots[i].data, linewidth=.2)
            plt.title(self.plots[i].name)
            plt.xlabel(self.plots[i].xLabel)
            plt.ylabel(self.plots[i].yLabel)
        plt.tight_layout()      #fix layout to make the plots appear nice
        plt.show()

    def run(self ):   #this function is called when the thread is started.
        self.PlotAllPlotsInOneWindow()


class Plot:   #simple data container for displaying plots

    def __init__(self, name, data, xLabel, yLabel):
        self.name = name
        self.data = data
        self.xLabel = xLabel
        self.yLabel = yLabel