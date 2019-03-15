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

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import os.path
import argparse
import sys

from datetime import datetime

parser = argparse.ArgumentParser(description='DQN agent')
parser.add_argument('-w', '--weight', help='Weight file name',default=None)
parser.add_argument('--train', action='store_true', help='Train agent')
parser.add_argument('--test', action='store_true', help='Test agent')
args = parser.parse_args()

nb_steps = 10
nb_episodes = 100
log_dir = './logs/DQN/log_{}'.format(datetime.now().strftime('%d-%m-%Y_%H:%M'))

if(args.test == True and args.weight == None):
    print("Provide the weight file name")
    sys.exit()
else:
    weight_dir = 'models/DQN/{}.h5f'.format(args.weight)

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
    dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, callbacks=[tbCallBack])

    dqn.save_weights('./models/DQN/{}_steps_{}_weights.h5f'.format(nb_steps,datetime.now().strftime('%d-%m-%Y_%H:%M')), overwrite=False)


#Test the model
def test():
    print("Testing model...")
    dqn.test(env, nb_episodes=nb_episodes, visualize=False)


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

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                value_min=.1, value_test=.05,nb_steps=1000000)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                    nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                    train_interval=4, delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    #Check previous models
    if (os.path.isfile(weight_dir)):
        print('Loading previous model...')
        dqn.load_weights(weight_dir)

    if (args.train):
        train()
    elif (args.test):
        test()
