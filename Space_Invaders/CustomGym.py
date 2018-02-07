"""
CustomGym
"""
import gym
import numpy as np
from scipy.misc import imresize # TODO: Use openCv resize

# Wrapper around Open AI Gym
class CustomGym:
    def __init__(self, game_name, skip_actions=4, num_frames=4, w=84, h=84):
        """
        skip_actions - The number of frames to repeat an action for
        num_frames - The number of frames stacked into one state to take an action
        """
        self.env = gym.make(game_name)
        self.num_frames = num_frames
        self.skip_actions = skip_actions
        self.w = w
        self.h = h

        if game_name == 'SpaceInvaders-v0':
            self.action_space=[1,2,3]

        self.action_size = len(self.action_space)
        self.state = None
        self.game_name = game_name

    # Process the input and stack the frames
    def preprocess(self, obs, is_start=False):
        # Convert to grayscale
        grayscale = obs.astype('float32').mean(2)
        # Resize and rescale to [0, 1]
        s = imresize(grayscale, (self.w, self.h)).astype('float32') * (1.0/255.0)
        # Reshape to a 4D array
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        # Stack frames, repeat num_frames if first frame
        # First frame
        if is_start or self.state is None:
            self.state = np.repeat(s, self.num_frames, axis=3)
        else:
            self.state = np.append(s, self.state[:, :, :, :self.num_frames-1], axis=3)
        return self.state

    def render(self):
        self.env.render()
    
    def reset(self):
        return self.preprocess(self.env.reset(), is_start=True)

    # make an action
    def step(self, action_idx):
        action = self.action_space[action_idx]
        accum_reward = 0
        prev_s = None

        for _ in range(self.skip_actions):
            s, r, term, info = self.env.step(action)
            accum_reward += r
            if term:
                break
            prev_s = s

        # Get around atari sprites flickering, take max value of current and previous frame
        if self.game_name == 'SpaceInvaders-v0' and prev_s is not None:
            s = np.maximum.reduce([s, prev_s])
        return self.preprocess(s), accum_reward, term, info
