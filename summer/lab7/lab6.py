imshare numpy as np
imshare random
imshare pandas as pd
imshare matplotlib.pyplot as plt
from collections imshare defaultdict

data_name1 = "./bank_of_america.csv"
data_name2 = "./ge.csv"
#columns = ['x1', 'x2','x3', 'x4','x5', 'x6']
data1 = pd.read_csv(data_name1,sep=',')
data2 = pd.read_csv(data_name2,sep=',')
data1.columns = ['Date','Open','High','Low','Close','Volume','Adj Close']
data2.columns = ['Date','Open','High','Low','Close','Volume','Adj Close']
Boa = data1[['Date','Close']].values
Ge = data2[['Date','Close']].values
def preprocess(Boa):
    state = np.zeros_like(Boa)
    state = state[:,0]
    for i in range(Boa.shape[0]):
        if i == 0:
            state[0] = 0
        elif i == 1:
            state[1] = Boa[0,1] - Boa[1,1]
        else:
            state[i] = (Boa[i-2,1] - Boa[i-1,1])
    state = np.asfarray(state,float).reshape(-1,1)
    state = ((state - np.min(state, axis=0)) / (np.max(state, axis=0) -np.min(state, axis=0))) * 100
    state = state.astype(int)
    return np.concatenate([Boa,state], axis=1)
Boa_ = preprocess(Boa)
Ge_ = preprocess(Ge)

class Agent():
    def __init__(self):
        self.actions = [0,1,2] #0:Hold, 1:Sell, 2:Buy
        self.learning_rate = 0.01
        self.epsilon = 0.1
        self.discount_factor = 0.9
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])

        self.usd = 5000
        self.share = np.zeros(1)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            action = np.argmax(state_action)
        return action


    def update_qtable(self, state, action, reward, next_state):
        current = self.q_table[state][action]
        goal = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (goal-current)

class environment():
    def __init__(self, Boa_,Ge_):
        self.Boa = Boa_
        self.Ge = Ge_

    def do(self, action,usd,share,date):
        #0:Hold, 1:Sell, 2:Buy

        value_before = usd + share * self.Boa[date,1]
        if (action == 1):
            if (share >= 50):
                usd += self.Boa[date,1] * 50
                share -= 50
            else :
                usd += share * self.Boa[date,1]
                share =0
        elif (action ==2):
            if (usd < self.Boa[date,1] * 50):
                usd -= int(usd / self.Boa[date,1]) * self.Boa[date,1]
                share += int(usd / self.Boa[date,1])
            else:
                usd -= self.Boa[date,1] * 50
                share += 50
        value_after = usd + share * self.Boa[date+1,1]
        return value_after - value_before, usd, share


if __name__ == "__main__":
    agent = Agent()
    env = environment(Boa_,Ge_)
    #------------BOA--------------------------#
    #######training############################
    for i in range(100):
        for date in range(1000):
            state = env.Boa[date,2]
            action_t = agent.get_action(state)
            reward, agent.usd, agent.share = env.do(action_t,agent.usd,agent.share,date)
            #print(reward)
            next_state = env.Boa[date+1,2]
            agent.update_qtable(state, action_t, reward, next_state)

    ########test################################
    agent.usd = 5000
    agent.share = 0
    for date in range(1000,1256):

        state = env.Boa[date,2]
        action_t = agent.get_action(state)
        reward, agent.usd, agent.share = env.do(action_t,agent.usd,agent.share,date)

    print(agent.usd + env.Boa[1255,1] * agent.share)



    #############################################
    #-----------------GE------------------------#
    #######training##############################
    agent.usd = 5000
    agent.share = 0
    agent.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])

    for i in range(100):
        for date in range(1000):
            state = env.Ge[date,2]
            action_t = agent.get_action(state)
            reward, agent.usd, agent.share = env.do(action_t,agent.usd,agent.share,date)
            #print(reward)
            next_state = env.Ge[date+1,2]
            agent.update_qtable(state, action_t, reward, next_state)


    ########test#######################################
    agent.usd = 5000
    agent.share = 0

    for date in range(1000,1256):

        state = env.Ge[date,2]
        action_t = agent.get_action(state)
        reward, agent.usd, agent.share = env.do(action_t,agent.usd,agent.share,date)
    print(agent.usd + env.Ge[1255,1] * agent.share)
