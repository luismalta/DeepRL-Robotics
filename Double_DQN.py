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
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import os.path


if __name__ == "__main__":
    env = gym.make('GazeboCircuit2cTurtlebotCameraNnEnv-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    env = wrappers.Monitor(env, '/tmp/{}'.format('teste'), force=True)

    np.random.seed(123)
    env.seed(123)
    nb_actions = 3


    model = Sequential()
    model.add(Flatten(input_shape=(1,) + (1,1,32,32)))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))

    model.add(Activation('linear'))
    print(model.summary())

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/Double_DQN', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    tbCallBack.set_model(model)

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   enable_double_dqn=True, target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    if (os.path.isfile('/home/luismalta/Projects/DeepRL/models/drl_GazeboCircuit2cTurtlebotCameraNnEnv-v0_Double_DQN_weights.h5f')):
        print('Loading previous model...')
        dqn.load_weights('/home/luismalta/Projects/DeepRL/models/drl_GazeboCircuit2cTurtlebotCameraNnEnv-v0_Double_DQN_weights.h5f')


    dqn.fit(env, nb_steps=200000, visualize=False, verbose=2, callbacks=[tbCallBack])


    dqn.save_weights('/home/luismalta/Projects/DeepRL/models/drl_{}_weights.h5f'.format('GazeboCircuit2cTurtlebotCameraNnEnv-v0_Double_DQN'), overwrite=True)
