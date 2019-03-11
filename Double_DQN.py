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

parser = argparse.ArgumentParser(description='Type of run: Train or Test')
parser.add_argument('-t', '--type', help='Type of run: Train or Test', required=True)
args = parser.parse_args()

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
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2, callbacks=[tbCallBack])

    dqn.save_weights('models/Double_DQN/drl_{}_weights.h5f'.format('Double_DQN'), overwrite=False)

#Test the model
def test():
    print("Testing model...")
    dqn.test(env, nb_episodes=100, visualize=False)


if __name__ == "__main__":

    if (args.type != "train" and args.type != "test"):
        print('Invalid Option')
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
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/Double_DQN', histogram_freq=0,
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
                    train_interval=4, delta_clip=1., enable_double_dqn=True)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    #Check previous models
    if (os.path.isfile('models/Double_DQN/drl_Double_DQN_weights.h5f')):
        print('Loading previous model...')
        dqn.load_weights('models/Double_DQN/drl_Double_DQN_weights.h5f')

    if (args.type == "train"):
        train()
    elif (args.type == "test"):
        test()
