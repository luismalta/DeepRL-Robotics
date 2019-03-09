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
    env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    env = wrappers.Monitor(env, '/tmp/{}'.format('teste'), force=True)

    np.random.seed(123)
    env.seed(123)
    nb_actions = 21

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Dense(300,input_shape=[1,100]))
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

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/DQN', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    tbCallBack.set_model(model)
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if (os.path.isfile('/home/luismalta/Projects/DeepRL/models/drl_GazeboCircuit2cTurtlebotCameraNnEnv-v0_DQN_weights.h5f')):
        print('Loading previous model...')
        dqn.load_weights('/home/luismalta/Projects/DeepRL/models/drl_GazeboCircuit2cTurtlebotCameraNnEnv-v0_DQN_weights.h5f')

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.test(env, nb_episodes=100, visualize=False)

    # After training is done, we save the final weights.
    #dqn.save_weights('/home/luismalta/Projects/DeepRL/models/drl_{}_weights.h5f'.format('GazeboCircuit2cTurtlebotCameraNnEnv-v0_DQN'), overwrite=False)
    #model.save_model('/home/luismalta/Projects/DeepRL/models/drl_{}_weights.h5f'.format('GazeboCircuit2cTurtlebotCameraNnEnv-v0_DQN')
