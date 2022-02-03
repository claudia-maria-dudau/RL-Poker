from poker_agent_general import PokerAgent

import tensorflow as tf
import collections
import numpy as np
import tqdm
from typing import List, Tuple
from enum import IntEnum
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.replay_mem = collections.deque(maxlen=buffer_size)

    # add a new experience to the buffer
    def add(self, state, action, reward, next_state, done):
        self.replay_mem.append((state, action, reward, next_state, done))

    # get a sample batch from the memory
    def sample(self, batch_size):
        if batch_size <= len(self.replay_mem):
            return random.sample(self.replay_mem, batch_size)
        else:
            assert False

    def __len__(self):
        return len(self.replay_mem)

class DQNModel(tf.keras.Model):
    def __init__(self, input_shape, n_action, method):
        super(DQNModel, self).__init__()

        # add an input layer of a given input shape
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)

        if method == MethodToUse.DQN_BASE:
            # avg 173.6 - 1000 ep

            # add a few hidden layers
            self.hidden_layers = []
            self.hidden_layers.append(tf.keras.layers.Dense(32, activation='tanh'))
            self.hidden_layers.append(tf.keras.layers.Dense(16, activation='relu'))
            self.hidden_layers.append(tf.keras.layers.Dense(16, activation='relu'))

            self.output_layer = tf.keras.layers.Dense(units=n_action, activation='linear')

        elif method == MethodToUse.DQN_TARGET_NETWORK:
            # avg 179 - 1100 ep

            # add a few hidden layers
            self.hidden_layers = []
            self.hidden_layers.append(tf.keras.layers.Dense(32, activation='relu'))
            self.hidden_layers.append(tf.keras.layers.Dense(32, activation='selu'))
            self.hidden_layers.append(tf.keras.layers.Dense(32, activation='relu'))

            self.output_layer = tf.keras.layers.Dense(units=n_action, activation='linear')

        elif method == MethodToUse.DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY:
            # 176.5 - 2700 ep

            # add a few hidden layers
            self.hidden_layers = []
            self.hidden_layers.append(tf.keras.layers.Dense(32, activation='tanh'))
            self.hidden_layers.append(tf.keras.layers.Dense(16, activation='selu'))
            self.hidden_layers.append(tf.keras.layers.Dense(8, activation='selu'))

            self.output_layer = tf.keras.layers.Dense(units=n_action, activation='linear')

        else:
            # avg 252.6 - 500 ep

            # add a few hidden layers
            self.hidden_layers = []
            self.hidden_layers.append(tf.keras.layers.Dense(32, activation='tanh'))
            self.hidden_layers.append(tf.keras.layers.Dense(16, activation='relu'))
            self.hidden_layers.append(tf.keras.layers.Dense(16, activation='relu'))

            self.output_layer = tf.keras.layers.Dense(units=n_action, activation='linear')

    @tf.function
    def call(self, inputs):
        # go through the input layer to ensure that inputs are of N x input_shape
        z = self.input_layer(inputs)

        # go over the hidden layers
        for l in self.hidden_layers:
            z = l(z)

        # results / state-action values are in the output layer
        q_values = self.output_layer(z)
        return q_values

class MethodToUse(IntEnum):
    DQN_BASE = 0
    DQN_TARGET_NETWORK = 1
    DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY = 2
    DDQN_AND_ALL = 3

class DQN_PokerAgent(PokerAgent):
    def __init__(self, env, seed = None, replay_buffer_size = 32000, gamma = 0.9,
                 batch_size = 1024, lr = 0.001, # learning rate (alpha)
                 steps_until_sync = 20, # at how many steps should we update the target network weights
                 pre_train_steps = 1, # steps to run before starting the training process
                 start_epsilon = 1, end_epsilon = 0.1, final_epsilon_step = 10000,
                 method = MethodToUse.DQN_TARGET_NETWORK, load_prev = False):
        super().__init__(method.name.lower(), env, gamma, lr, start_epsilon, end_epsilon, (start_epsilon - end_epsilon) / (final_epsilon_step - pre_train_steps))

        self.method = method

        # model for function approximation methods
        observation_space_shape = list(env.observation_space)
        self.dqn = DQNModel(input_shape=observation_space_shape, n_action=self.env.action_space.n, method=self.method)

        # input = BatchSize (unknown, that's why it is None) X input_shape of the environment
        inputShape = [None]
        inputShape.extend(observation_space_shape)
        self.dqn.build(tf.TensorShape(inputShape))

        if self.method > MethodToUse.DQN_BASE:
            self.dqn_target = DQNModel(input_shape=observation_space_shape, n_action=env.action_space.n, method=self.method)
            self.dqn_target.build(tf.TensorShape(inputShape))

        # replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size)

        # all other parameters
        self.batch_size = batch_size
        self.steps_until_sync = steps_until_sync
        self.pre_train_steps = pre_train_steps
        self.final_epsilon_step = final_epsilon_step

        # loss function
        self.loss_function = tf.keras.losses.MSE

        # optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

        # deterministic output at each run for easy debugging
        if seed is not None:
            self.env.seed(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # total steps of the running algorithm
        self.total_steps = 0

        if load_prev:
            self.load_model()

    # get epsilon
    def get_epsilon(self, no_episode):
        if no_episode <= self.pre_train_steps:
            return 1.0 # full exploration in the beginning
        else:
            epsilon = max((1.0 - self.epsilon_decay * (no_episode - self.pre_train_steps)), self.end_epsilon)
            return epsilon

    # receive action returns a state, reward, done (wrapped to be compatible in a tf graph op)
    def env_step(self, action : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.float32),
                np.array(done, np.float32))

    # this is under a tensorflow wrapper to be able to use it inside tensorflow graphs
    def tf_env_step(self, action:tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.float32])

    # gets an action based on epsilon and the DQN model
    def select_action(self, states, epsilon):
        # current legal moves
        legal_moves = [x.value for x in self.env.legal_moves]

        # random action choice when in the pre-training phase, or we still wait to fill 
        # some data in the buffer, or sampled value is less than epsilon
        if self.total_steps <= self.pre_train_steps or \
            np.random.rand() < epsilon or len(self.replay_buffer) < self.batch_size:

            actions = np.random.choice(legal_moves, size=[len(states), ])

        else:
            # run the network and get the action to do for each of the states in the batch
            predict_q = self.dqn(states)

            # get the argmax on the columns
            actions = np.argmax(predict_q, axis=1)

        return actions

    # runs a single episode to collect training data
    # returns the reward from this episode, the average loss and epsilon, avg magnitude of grads for each layer
    def run_episode(self) -> Tuple[float, float, float, tf.Tensor]:
        # grab the initial state and convert it to a tensorflow type tensor
        initial_state = self.env.reset()
        initial_state = tf.constant(initial_state, dtype=tf.float32)

        initial_state_shape = initial_state.shape
        current_state = initial_state
        current_action = None
        episode_reward = tf.constant(0.0, dtype=tf.float32)
        total_loss = 0.0

        # accumulate the total gradient magnitude on each trainable variable
        total_grads_magnitude = tf.constant(0.0, shape=[len(self.dqn.trainable_variables),])

        no_losses_computed = 0
        epsilon = 1.0

        for episode_step in tf.range(self.max_steps_per_episode):
            # get epsilon for this step
            epsilon = self.get_epsilon(no_episode=self.total_steps)

            # convert to a batch 1 tensor the state
            current_state_batched = tf.expand_dims(current_state, 0)

            actions = self.select_action(current_state_batched, epsilon)
            current_action = tf.constant(actions[0]) # first action in the batch

            # apply action, get next step and reward
            new_state, reward, done = self.tf_env_step(current_action)
            new_state.set_shape(initial_state_shape)
            episode_reward += reward

            # add the experience to the replay buffer
            self.replay_buffer.add(state=current_state, action=current_action, reward=reward, next_state=new_state, done=done)

            # check if we should do a training step: did we past the pre train phase 
            # and does the replay buffer contain a certain replay buffer size ?
            if (self.total_steps > self.pre_train_steps and len(self.replay_buffer) >= self.batch_size):
                loss, grads_magnitude = self.train_step()
                total_loss += loss
                total_grads_magnitude += grads_magnitude
                no_losses_computed += 1

            # check if we should update the target network weights
            if self.method > MethodToUse.DQN_BASE and self.total_steps % self.steps_until_sync == 0:
                self.dqn_target.set_weights(self.dqn.get_weights())

            # update the current state, total episodes
            current_state = new_state
            self.total_steps += 1

            # check if episode ended
            if done == 1.0:
                break

        # calculating average loss and gradients magnitude
        avg_loss = (total_loss / no_losses_computed if no_losses_computed > 0 else 0)
        avg_grads_magnitude = (total_grads_magnitude / no_losses_computed if no_losses_computed > 0 else 0)

        return episode_reward, avg_loss, epsilon, avg_grads_magnitude

    # runs a train step using the current replay buffer
    # returns the loss and gradients magnitude for updating the model
    def train_step(self):
        # sample a batch from replay memory
        train_batch = self.replay_buffer.sample(batch_size=self.batch_size)

        # get separate tensors for states, actions, rewards, nextstates, dones
        b_states = tf.stack([x[0] for x in train_batch], axis=0)
        b_actions = tf.stack([tf.cast(x[1], tf.int32) for x in train_batch], axis=0)
        b_rewards = tf.stack([x[2] for x in train_batch], axis=0)
        b_next_states = tf.stack([x[3] for x in train_batch], axis=0)
        b_dones = tf.stack([x[4] for x in train_batch], axis=0)

        # perfom loss computations under an automatic gradient computation scope
        with tf.GradientTape() as tape:
            # get the list of trainable variables and make them watchable by the gradient tape
            dqn_variables = self.dqn.trainable_variables
            tape.watch(dqn_variables)

            # compute the estimated values from next_states batch Q(nextState, action) for all actions
            q_next_state = self.dqn(b_next_states) if self.method <= MethodToUse.DQN_BASE \
                                                else self.dqn_target(b_next_states)

            # compute the next best action to select from q_next_state (reduce on the columns axis in the batch)
            next_best_actions = tf.argmax(q_next_state, axis=1)

            # compute the targetQ value
            b_next_best_actions_one_hot_encoding = tf.one_hot(indices=next_best_actions, depth=self.n_action)
            targetQ = b_next_best_actions_one_hot_encoding * q_next_state
            targetQ = tf.reduce_sum(targetQ, axis=1)

            # for each element in the batch: 
            # if done is true, then the reward is considered final
            # if not, then add the estimation from next state too
            targetQ = b_rewards + (1.0 - b_dones) * self.gamma * targetQ

            # get the estimates for each state in the batch
            Q_states = self.dqn(b_states)

            # compute the predictedQ (Q(state, action) for each state-action pair in the training batch)
            b_actions_one_hot_encoding = tf.one_hot(indices=b_actions, depth=self.n_action) # (a)
            predictedQ = b_actions_one_hot_encoding * Q_states
            predictedQ = tf.reduce_sum(predictedQ, axis=1)

            # apply loss function
            loss = self.loss_function(y_true=targetQ, y_pred=predictedQ)

        # backpropagation: compute the gradients from the loss and apply them using the optimizer
        grads = tape.gradient(loss, dqn_variables)
        self.optimizer.apply_gradients(zip(grads, dqn_variables))

        # compute the gradients magnitude to be sure that we are still able to learn something
        grads_magnitude = tf.convert_to_tensor([tf.norm(layer) for layer in grads])

        return loss, grads_magnitude

    def train(self, max_episodes, max_steps_per_episode, reward_threshold, no_episodes_for_average = 10):
        episode_reward_history = []
        last_rewards = collections.deque(maxlen=no_episodes_for_average)

        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

        with tqdm.trange(self.max_episodes) as t:
            for episode in t:
                # run a full episode and get the reward from it
                episode_reward, avg_loss, epsilon_used, grads_magnitude = self.run_episode()

                # update statistics from this episode
                episode_reward_history.append(episode_reward)
                last_rewards.append(episode_reward)
                mean_episode_reward = np.mean(last_rewards)

                t.set_description(f'Episode {episode}')
                t.set_postfix(episode_reward=episode_reward, running_reward=mean_episode_reward)
                if episode % no_episodes_for_average == 0:
                    print(f'\nEpisode {episode} (total step) {self.total_steps}: average reward: {mean_episode_reward}. avg loss: {avg_loss} epsilon used: {epsilon_used} grads mag: {grads_magnitude}')
                    
                    if episode != 0 and mean_episode_reward > self.max_score[self.method.value]:
                        self.max_score[self.method.value] = mean_episode_reward
                        self.round_max_score[self.method.value] = episode
                        self.save_model()
                    
                    print(self.max_score[self.method.value], self.round_max_score[self.method.value])

                if episode != 0 and mean_episode_reward > reward_threshold:
                    break

        print(f"\nSolved at episode{episode}: average rewards {mean_episode_reward:.2f}!")
        print(self.max_score[self.method.value], self.round_max_score[self.method.value])

        self.save_model()

    def save_model(self):
        self.dqn.save_weights(self.name)

    def load_model(self):
        print(self.name)
        self.dqn.load_weights(self.name)
        self.dqn_target.set_weights(self.dqn.get_weights())

    def play(self, no_episodes):
        scores = []
        actions_per_ep = []

        for episode_index in tqdm.trange(no_episodes):
            # for the record
            score = 0
            no_actions = 0

            # initiate state
            state = tf.constant(self.env.reset(), dtype=tf.float32)
            state_shape = state.shape

            while True:
                # convert to a batch 1 tensor the state
                state_batched = tf.expand_dims(state, 0)

                actions = self.select_action(state_batched, self.end_epsilon)
                current_action = tf.constant(actions[0]) # first action in the batch

                # apply action, get next step and reward
                next_state, reward, done = self.tf_env_step(current_action)
                next_state.set_shape(state_shape)

                # records
                score += reward
                no_actions += 1

                # move to next state
                state = next_state

                if done:
                    break

            # record
            scores.append(score)
            actions_per_ep.append(no_actions)

        average_score = np.mean(scores)
        average_actions = np.mean(actions_per_ep)

        print(f'Average score: {average_score}, Average actions per ep: {average_actions}')

        return average_score, average_actions