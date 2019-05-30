import tensorflow as tf
import numpy as np
from collections import deque
import random
import gym
from gym import wrappers
import gym_gazebo
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import keras

import json

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import os.path
import os
import errno
import argparse
import sys

from datetime import datetime

parser = argparse.ArgumentParser(description='DQN agent')
parser.add_argument('-w', '--weight', help='Weight file name',default=None)
parser.add_argument('--train', action='store_true', help='Train agent')
parser.add_argument('--test', action='store_true', help='Test agent')
args = parser.parse_args()

params = {
    'train_test' : {
        'nb_steps': 1000000,
        'nb_episodes_test' : 100,
        'time_start': datetime.now().strftime('%d-%m-%Y_%H:%M')
    },
    'agent' : {
        'nb_steps_warmup':5000,
        'gamma':.99,
        'target_model_update':10000,
        'train_interval':4,
        'delta_clip':1
    },
    'police':{
        'exploretion_value_max':1.,
        'exploretion_value_min':.1,
        'exploretion_value_test':.05,
        'exploration_nb_steps':250000
    },
    'compile':{
        'learn_rate':0.00025,
        'metrics':['mae']
    }
}


log_dir = './logs/DQN/log_{}'.format(params['train_test']['time_start'])
import distutils.dir_util
distutils.dir_util.mkpath(log_dir)

if(args.test == True and args.weight == None):
    print("Provide the weight file name")
    sys.exit()
else:
    weight_dir = args.weight

with open(log_dir +'/params.json', 'w') as fp:
    json.dump(params, fp)




#Build the model for the agent
def build_model():
    model = Sequential()
    model.add(Dense(300, input_shape=[1, 100]))
    model.add(Activation('linear'))
    model.add(Flatten())
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    return model

#Train the model
def train():
    print("Training model...")
    train_history = dqn.fit(env, nb_steps=params['train_test']['nb_steps'], visualize=False, verbose=2, callbacks=[tbCallBack])
    print(train_history)

    with open(log_dir + '/train_history.json', 'w') as fp:
        json.dump(train_history.history, fp)

    dqn.save_weights('{}/{}_weights.h5f'.format(log_dir,params['train_test']['time_start']), overwrite=False)


#Test the model
def test():
    print("Testing model...")
    test_history = dqn.test(env, nb_episodes=params['train_test']['nb_episodes_test'], visualize=False)

    with open(log_dir + '/test_history.json', 'w') as fp:
        json.dump(test_history.history, fp)

if __name__ == "__main__":

    if (args.train == False and args.test == False):
        print('No flag was passed')
        sys.exit()

    #Env setup
    env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')

    outdir = '/tmp/gazebo_gym_experiments/'
    env = wrappers.Monitor(env, '/tmp/{}'.format('teste'), force=True)

    np.random.seed(123)
    env.seed(123)
    nb_actions = 21

    model = build_model()

    #Tensorboard callback
    tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                        batch_size=32, write_graph=True, write_grads=False,
                                        write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                                        embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    tbCallBack.set_model(model)

    #Agent configuration
    memory = SequentialMemory(limit=50000, window_length=1)

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=params['police']['exploretion_value_max'],
                                value_min=params['police']['exploretion_value_min'], value_test=params['police']['exploretion_value_test'],
                                nb_steps=params['police']['exploration_nb_steps'])

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                    nb_steps_warmup=params['agent']['nb_steps_warmup'], target_model_update=params['agent']['target_model_update'],
                    train_interval=params['agent']['train_interval'], delta_clip=params['agent']['delta_clip'],gamma=params['agent']['gamma'])

    dqn.compile(Adam(lr=params['compile']['learn_rate']), metrics=params['compile']['metrics'])

    #Check previous models
    if (os.path.isfile(weight_dir)):
        print('Loading previous model...')
        dqn.load_weights(weight_dir)

    if (args.train):
        train()
    elif (args.test):
        test()
