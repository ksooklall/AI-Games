"""
Applying A3C to space invaders
https://cgnicholls.github.io/reinforcement-learning/2017/03/27/a3c.html
"""
import threading
from queue import Queue

import gym
import numpy as np
import tensorflow as tf
from time import time, sleep

from Agent import Agent
from CustomGym import CustomGym
from Summary import Summary

def sample_game(env, n):
    env.reset()
    for i in range(n):
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
    env.close()

# Trainer to work on a single thread
def async_trainer(agent, env, sess, thread_idx, T_queue, summary, saver, save_path):
    """
    agent - Worker (class)
    env - Env object (object)
    sess - current working session (tf object)
    thread_idx - Current thread (int)
    T_queue - Iteration counter
    summary -
    saver - save current weights (tf object)
    save_path - path to saved model (str)
    """
    
    print(" Training thread: {}".format(thread_idx))
    T = T_queue.get()
    T_queue.put(T+1)
    last_verbose = T
    last_time = time()
    last_target_update = T

    # Starting point
    terminal = True
    while T < T_MAX:
        batch_states = []
        batch_rewards = []
        batch_actions = []
        baseline_values = []

        if terminal:
            terminal = False
            state = env.reset()

        while not terminal and len(batch_states) < I_ASYNC_UPDATE:
            # Save the current state
            batch_states.append(state)

            # Choose an action randomly according to the policy prob
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)

            # Take the action and get the next state, reward and terminal
            state, reward, terminal, info = env.step(action_idx)

            # Update counter
            T = T_queue.get()
            T_queue.put(T+1)

            # Clip reward to be between -1 and 1
            reward = np.clip(reward, -1, 1)

            # Save rewards and actions
            batch_rewards.append(reward)
            batch_actions.append(action_idx)
            baseline_values.append(value[0])

        target_value = 0
        # If last state was terminal R = 0 else get the last estimated value of the last state
        if not terminal:
            target_value = agent.get_value(state)[0]
        last_R = target_value

        # Compute the sampled n-step discounted reward
        batch_target_values = []
        for reward in batch_rewards[::-1]:
            target_value = reward + DISCOUNT_FACTOR * target_value
            batch_target_values.append(target_value)

        batch_target_values.reverse()

        # Compute the estimated value of each state
        batch_advantages = np.array(batch_target_values) - np.array(baseline_values)
        
        # Compute the asynchronous gradients update
        agent.train(np.vstack(batch_states), batch_actions, batch_target_values, batch_advantages)

    global training_finished
    training_finished = True

# Asynchronous run first
def a3c(game_name, num_threads=8, restore=None, save_path=None):

    # A list of environments
    envs = [CustomGym(game_name) for _ in range(num_threads)]
    # Evaluation environment 
    evaluation_env = CustomGym(game_name)

    # Create session to use
    with tf.Session() as sess:
        # Create agent
        agent = Agent(sess, action_size=envs[0].action_size,
                      optimizer=tf.train.AdamOptimizer(INITIAL_LEARNING_RATE))
        
        # Create saver, keeping two checkpoints
        saver = tf.train.Saver(max_to_keep=2)
        T_queue = Queue()

        # Restore or create parameters, put worker to work
        if restore is not None:
            saver.restore(sess, save_path + '-' + str(restore))
            last_T = restore
            print("T was: {}".format(last_T))
            T_queue.put(last_T)
        else:
            sess.run(tf.global_variables_initializer())
            T_queue.put(0)
        summary = Summary(save_path, agent)

        # Create a process for each agent
        processes = []
        for i in range(num_threads):
            processes.append(threading.Thread(target=async_trainer, args=(agent, envs[i], sess, i,
                                                                    T_queue, summary, saver, save_path,)))
        # Create a process to evaluate the agent
        processes.append(threading.Thread(target=evaluator, args=(agent, evaluation_env, sess,
                                                                  saver, summary, save_path, T_queue,)))
        
        # Once all threads are done, kill them (daemon)
        for p in processes:
            p.daemon = True
            p.start()
        # Until training is finished
        while not training_finished:
            sleep(0.01)
        # After thread terminates join so we get the thread back
        for p in processes:
            p.join()

# Evaluator function to evaluate every training_step
def evaluator(agent, env, sess, saver, summary, save_path, T_queue):
    T = T_queue.get()
    T_queue.put(T)
    last_time = time()
    last_verbose = T
    while T < T_MAX:
        T = T_queue.get(T)
        T_queue.put(T)
        if T - last_verbose >= VERBOSE_EVERY:
            print('T: {}'.format(T))
            current_time = time()
            print('Train steps per second: {}'.format(float(T - last_verbose) / (current_time - last_verbose)))
            last_time = current_time
            last_verbose = T

            print('Evaluating agent')
            episode_rewards, episode_vals = estimate_reward(agent, env, episodes=5)
            avg_ep_r = np.mean(episode_rewards)
            avg_value = np.mean(episode_vals)
            print('Avg ep reward: {} Avg value: {}'.format(avg_ep_r, avg_value))

            summary.write_summary({'episode_avg_reward': avg_ep_r, 'avg_value': avg_value}, T)
            checkpoint_file = saver.save(sess, save_path+FILE_NAME, global_step=T)
            print('Saved in: {}'.format(checkpoint_file))
        sleep(1.0)

# Estimate the reward of the agent at a given point
def estimate_reward(agent, env, episodes=10, max_steps=10000):
    episode_rewards = []
    episode_vals = []
    steps = 0
    for i in range(episodes):
        episode_reward = 0
        state = env.reset()
        terminal = False
        while not terminal:
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, _ = env.step(action_idx)
            steps += 1
            episode_vals.append(value)
            episode_reward += reward

            if steps > max_steps:
                episode_rewards.append(episode_reward)
                return episode_rewards, episode_vals
        episode_rewards.append(episode_reward)
        
    return episode_rewards, episode_vals

def run_agent(save_path, game_name):
    with tf.Session() as sess:
        agent = Agent(sess, action_size=3)
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
        
        play(agent, game_name)
        return sess, agent

# View the agent playing
def play(agent, game_name, render=True, num_episodes=10, fps=5.0, monitor=False):
    gym_env = gym.make(game_name)
    if monitor:
        gym.wrappers.Monitor(gym_env, directory='videos/-v0', video_callable=lambda episode_id: True)
    env = CustomGym(game_name)

    desired_frame_length = 1.0/fps

    episode_rewards = []
    episode_vals = []
    for ep in range(num_episodes):
        print('Starting episode: {}'.format(ep))
        episode_reward = 0
        state = env.reset()
        terminal = False
        current_time = time()
        while not terminal:
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, _ = env.step(action_idx)
            if render:
                env.render()
            episode_vals.append(value)
            episode_reward += reward
            frame_length = time() - current_time
            if frame_length < desired_frame_length:
                sleep(desired_frame_length - frame_length)
            current_time = time()
        episode_rewards.append(episode_reward)
    if monitor:
        gym_env.monitor.close()
    return episode_rewards, episode_vals

if __name__ == '__main__':
    #sample_game(env, 1000)

    # Train for this many time steps
    T_MAX = 10000000
    # Use this many threads
    NUM_THREADS = 8
    # Initial learning rate
    INITIAL_LEARNING_RATE = 1e-4
    # Discount Factor
    DISCOUNT_FACTOR = 0.99
    # Evaluate the agent and print out average reward fixed time steps
    VERBOSE_EVERY = 40000
    # Update the parameters in each thread after this many steps in that thread
    I_ASYNC_UPDATE = 5
    # Use this global variable to exit the training loop in each thread once we've finished
    training_finished = False
    FILE_NAME = str(NUM_THREADS) + '-thread.tf'

    #a3c(game_name='SpaceInvaders-v0', num_threads=NUM_THREADS, save_path='log/')
    run_agent('log/{}'.format(FILE_NAME)+'-9984608', game_name='SpaceInvaders-v0')
