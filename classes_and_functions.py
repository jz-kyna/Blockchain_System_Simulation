'''This file includes all classes and functions/methods created for the 5580 final project.'''
'''It will be saved as a .py file and can be imported for simulation purpose.'''

##Import Libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

##Helping classes
class Empty: 
    '''create an error Class'''
    pass

class Transaction:
    '''Class of Transaction object:
    
       start: arrival time of each transaction
       percentage: transaction fee percentatge for each transaction
       amount: tranfer amount for each transaction
       fee: transaction fee for each transaction'''
    
    def __init__(self,amount,percentage,start):
        self.starttime = start
        self.amount = amount
        self.percentage = percentage
        self.fee = amount*percentage
        self.endtime = 0
        
    def __repr__(self):
        '''representative/print: starttime of transaction'''
        return str(self.starttime)
      
    def __lt__(self,other):
        '''comparison：transaction.fee'''
        return self.fee < other.fee
           
class PriorityQueueBase:
    """Class for the blockchain system"""
    
    def __init__(self, name='', arrivalRate = 120, minerRate=1, blockSize=1):
        self.name = name
        self.arrivalRate = arrivalRate  #lmbda
        self.minerRate = minerRate      #mu
        self.blockSize = blockSize      #K
        self.waitingentities = []       #queue for transactions waiting for mining
        self.inBlock = []               #transactions being mining
        self.minerOn = 0                #whether the miner is working or not
        
    def __repr__(self):
        queue = self.name + " queue: " + str(self.waitingentities) 
        system = self.name + "InBlock: " + str(self.inBlock) 
        return queue + " \n " + system    
    
    def is_empty(self):
        return len(self.waitingentities) == 0
    
    def arrive(self, newarrival):
        self.waitingentities.append(newarrival)
    
    def startMining(self):
        if self.is_empty():
            raise Empty
        else:
            self.waitingentities.sort(reverse = True)
            entities = self.waitingentities[0:self.blockSize]
            self.inBlock.append(entities)
            self.waitingentities = self.waitingentities[self.blockSize:]
            self.minerOn = 1
            
    def endMining(self):
        leavingTransactions = self.inBlock.pop()
        self.minerOn = 0
        return leavingTransactions
        

##Simulation class
class SimulationBasic():
    '''The SimulationBasic class takes inputs and runs the simulation under assumptions for the basic model.
       
       Inputs:
       minerRate_list: list, values/value of mu
       blockSize_list: list, corresponding values/value of K, given mu as minerRate, lmbda = 120 and the system is in steady state
       t_end: runtime of each single simulation
       results: list of metrics collected during each simulation
       '''
    def __init__(self, minerRate_list,blockSize_list,t_end):
        self.t_end = t_end
        self.arrivalRate = 120
        self.minerRate_list = minerRate_list
        self.blockSize_list = blockSize_list
        
        #store the simulation results
        self.results = []
        
    def event(self, minerRate, arrivalRate, minerOn):
        ## minerOn in {0,1}
        total_rate = minerRate*minerOn + arrivalRate
        U = np.random.rand()
        if U <= arrivalRate/total_rate:
            event = 'transArrival'
        else:
            event = 'endMining' 
        return event
    
    def oneTimeSimulate(self,minerRate,arrivalRate,blockSize,fixed_amount,percentage_list):
        """simulation for a pair of fixed mu, k and lambda.
        
           Inputs: 
           fixed_amount: if it's not equal to 0, then fix the transaction amount as fixed_amount for each transaction arrives later
           percentage_list: an percentage list used to initilize the transaction fee percentage for each transaction
           """
        minerOn = 0
        totalRate = minerRate*minerOn+arrivalRate
        blockchain = PriorityQueueBase(minerRate = minerRate, blockSize = blockSize) #Set up the blockchain system
        
        ## time log & next event time generation
        t = 0
        nextEventTime = t + np.random.exponential(1/totalRate)
        
        ## track of metrics
        times = []
        
        finished_trans = []    # list, Transactions finished mining/in the block chain
        num_transactions = []  # list, total number of Transactions in the block chain per unit time
        totalBRC = []          # list, total transaction amount in the block chain per unit time
        rateofFee = []         # list, total transaction fees in the block chain per unit time
        lengthofQueue = []     # list, number of transactions waiting in the queue
        delay_list = []        # list, delay time of each Transactions added to the block chain
        delay = []             # list, average pending time of Transactions
                
        ## start simulation
        while t < self.t_end: 
          
            t = nextEventTime
            lengthofQueue.append(len(blockchain.waitingentities)/t)
            event = self.event(minerRate,arrivalRate,blockchain.minerOn)
            
            if event == 'transArrival':
                if fixed_amount:
                  amount = fixed_amount
                else:
                  amount = np.random.randint(5,25+1)
                percentage = np.random.choice(percentage_list,1)[0] 
                newArrival = Transaction(amount,percentage,t)         ## Initiate a new transaction
                blockchain.arrive(newArrival)
                
                if blockchain.minerOn == 0 and len(blockchain.waitingentities) >= blockchain.blockSize:
                    blockchain.startMining()   
                
            elif event == 'endMining':
                leavingTransactions = blockchain.endMining()
                finished_trans += leavingTransactions                 ## Record transactions added to the block chain
                for leavingTrans in leavingTransactions:
                    leavingTrans.endtime = t
                delay_list += [t-trans.starttime for trans in leavingTransactions]
                if len(blockchain.waitingentities) >= blockchain.blockSize:  
                    blockchain.startMining()
                    
            totalRate = minerRate*blockchain.minerOn+arrivalRate
            nextEventTime = t + np.random.exponential(1/totalRate)
            
            ## Store the metrics
            times.append(t)
            if len(finished_trans) != 0:  
                num_transactions.append(len(finished_trans)/t)
                totalBRC.append(sum([trans.amount for trans in finished_trans])/t)
                delay.append(np.mean(delay_list))
                rateofFee.append(sum([trans.fee for trans in finished_trans])/t)
            else:
                num_transactions.append(0)
                totalBRC.append(0)
                delay.append(0)
                rateofFee.append(0)            
        return [times,lengthofQueue,num_transactions,totalBRC,delay,rateofFee,finished_trans]  
    
    def multiTimeSimulate(self):
        """Simulte for each combination of mu and k with fixed arrival rate and store the results in self.results"""
        
        for minerRate in self.minerRate_list:
            for blockSize in self.blockSize_list:
                print('The simulation for minerRate = {} and blockSize = {} starts.'.format(minerRate,blockSize))
                p_list = [0.01,0.02]
                result = self.oneTimeSimulate(minerRate = minerRate,arrivalRate = self.arrivalRate,blockSize = blockSize,fixed_amount = 0,percentage_list = p_list)
                result.append((minerRate,blockSize))
                self.results.append(result) 
    
    def fixedSimulate(self,fixed_minerRate,fixed_blockSize,amount,p_list): 
        '''simulate the relation between transaction fee and the delay by setting up the transfer amount of 
        each transaction with a fixed transfer amount, and vary the transaction fee through the percentage'''
                              
        result = self.oneTimeSimulate(minerRate=fixed_minerRate,arrivalRate = self.arrivalRate,blockSize = fixed_blockSize,fixed_amount = amount,percentage_list = p_list)
        finished_trans = result[-1]
        transfee_type = [amount*p for p in p_list]
        delay = np.zeros(len(transfee_type))
        for trans in finished_trans:
            index = transfee_type.index(trans.fee)
            delay[index] += (trans.endtime-trans.starttime)/len(finished_trans) 
        return [transfee_type,delay]    

def steadyState(results, t_warmup, muList, kList):
  """Complie the multiple simulation results in SimulationBasic Class"""
  
  steadyData = []
  n = len(results)
  n_mu = len(muList)
  n_k = len(kList)
  
  lengthofQueue = np.zeros(n)
  Trans = np.zeros(n)
  BRC = np.zeros(n)
  Delay = np.zeros(n)
  RateofFee = np.zeros(n)
  
  for i in range(n):
    simResult = results[i]
    mu, k = simResult[7]
    times = np.array(simResult[0])
    ind = (times >= t_warmup)
    print("When minerRate = {} and blockSize = {}, the steady states are shown below:".format(mu,k))
    
    #steady_times = times[ind]
    lengthofQueue[i] = np.mean(np.array(simResult[1])[ind])
    Trans[i] = np.mean(np.array(simResult[2])[ind])
    BRC[i] = np.mean(np.array(simResult[3])[ind])
    Delay[i] = np.mean(np.array(simResult[4])[ind])
    RateofFee[i] = np.mean(np.array(simResult[5])[ind])    
        
    #steadyData.append([steady_trans,steady_BRC,steady_delay,steady_rateofFee])
    print("Length of Queue = {}; Rate of Transaction = {} ;Rate of BRC = {}; Delay = {}; Rate of Fee = {}.\n".format(lengthofQueue[i],Trans[i],BRC[i],Delay[i],RateofFee[i]))
  lengthofQueueTab = pd.DataFrame((lengthofQueue.reshape(n_mu,n_k)).T, index = kList, columns = muList)
  TransTab = pd.DataFrame((Trans.reshape(n_mu,n_k)).T, index = kList, columns = muList)
  BRCTab = pd.DataFrame((BRC.reshape(n_mu,n_k)).T, index = kList, columns = muList)
  DelayTab = pd.DataFrame((Delay.reshape(n_mu,n_k)).T, index = kList, columns = muList)
  RateofFeeTab = pd.DataFrame((RateofFee.reshape(n_mu,n_k)).T, index = kList, columns = muList)
  
  steadyData = [lengthofQueueTab,TransTab,BRCTab,DelayTab,RateofFeeTab]
  
  return steadyData