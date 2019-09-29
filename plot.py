# -*- coding: utf-8 -*-
#!/usr/bin/env python

import matplotlib.pyplot as plt
import os.path
import os
import argparse
import json



parser = argparse.ArgumentParser(description='Train report build')
parser.add_argument('-a', '--history', help='History file name',default=None)
args = parser.parse_args()

history_list = []
model_cont = 1

def search_history(*args):
    print()
    for item in args:
        for p, _, files in os.walk(os.path.abspath(item)):
            for file in files:
                if(file == 'train_history.json'):
                    with open(os.path.join(p, file), 'r') as fp:
                        train_history = {
                            "history" : json.load(fp),
                            "label" : str(p)
                        }

                        history_list.append(train_history)


def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed

search_history("/home/luismalta/Projects/DeepRL/logs/Dueling_DQN")



# with open('/home/luismalta/Projects/DeepRL/logs/Dueling_DQN/log_01-06-2019_22:28/train_history.json') as fp:
#     train_history = {
#         "history" : json.load(fp),
#         "label" : 'DUELING DQN MODELO 7'
#     }
#     history_list.append(train_history)
#
# with open('/home/luismalta/Projects/DeepRL/logs/Dueling_DQN/log_04-07-2019_00:51/train_history.json') as fp:
#     train_history = {
#         "history" : json.load(fp),
#         "label" : 'DUELING DQN MODELO 4'
#     }
#     history_list.append(train_history)
#
# with open('/home/luismalta/Projects/DeepRL/logs/Dueling_DQN/log_10-07-2019_23:44/train_history.json') as fp:
#     train_history = {
#         "history" : json.load(fp),
#         "label" : 'DUELING DQN MODELO 8'
#     }
#     history_list.append(train_history)


plt.rcParams['figure.figsize']= (15,5)
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.weight':0})
plt.get_cmap()


for history in history_list:
    reward = smooth(history["history"]["episode_reward"],0.995)
    #plt.plot(reward, label=history["label"][-21:])
    plt.plot(reward, label="DUELING DQN MODELO " + str(model_cont))
    model_cont += 1


plt.legend()
plt.grid()
plt.title(u'Recompensas por epsódios', loc='left', fontsize=16, fontweight=10, color='black')
plt.xlabel(u"Episódios")
plt.ylabel("Recompensa")
plt.show()
