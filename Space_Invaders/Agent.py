import tensorflow as tf
"""
The agent
"""
class Agent():
    def __init__(self, session, action_size, optimizer=tf.train.AdamOptimizer(1e-4)):
        self.action_size = action_size
        self.sess = session
        self.optimizer = optimizer

        with tf.variable_scope('network'):
            # Give the agent the state, policy and value from the model
            self.state, self.policy, self.value = self.build_model(84, 84, 4)

            # Get weights for the network
            self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')

            # agent action, advantage and target value
            self.action = tf.placeholder(tf.int32, [None], name='action')
            self.target_value = tf.placeholder('float32', [None], name='target_value')
            self.advantages = tf.placeholder('float32', [None], name='advantages')
            
        with tf.variable_scope('optimizer'):
            # One hot encode the actiosn
            action_one_hot = tf.one_hot(self.action, self.action_size)
            # Clip the policy output to avoid zeros and ones
            min_policy = 0.00001
            max_policy = 0.99999
            self.log_policy = tf.log(tf.clip_by_value(self.policy, min_policy, max_policy))
            # For a given state and action compute the log of the policy at that action for that state, for that batch
            self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, action_one_hot),
                                                   reduction_indices=1)

            # Loss Functions for policy and value
            # The negative sign is for gradient ascent, self.policy domain (0, 1)
            self.policy_loss = -tf.reduce_mean(self.log_pi_for_action * self.advantages)
            self.value_loss = tf.reduce_mean(tf.square(self.target_value - self.value))
            # Entropy as another loss to the policy from Mnih's paper
            # The entropy of the policy is the expected_value of the -log(P(x)) -> E(-log(P(x))
            # larger entropy implies more exploration
            # small entropy is when the probability distribution is more concentrated
            self.entropy = tf.reduce_sum(tf.multiply(self.policy, -self.log_policy))
            

            # Minimize loss, subtract entropy to encourage exploration
            # MODIFIABLE - 0.5, 0.01
            self.loss = 0.5 * self.policy_loss + self.value_loss - self.entropy * 0.01

            # Compute the gradients of all the Loss wrt all the weights
            grads = tf.gradients(self.loss, self.weights)
            # MODIFIABLE - Clipping ratio - 40.0
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_vars = list(zip(grads, self.weights))

            # Apply the gradients to the optimizer
            self.train_optimizer = optimizer.apply_gradients(grads_vars)

    # Get value from V(s)
    def get_value(self, state):
        return self.sess.run(self.value, {self.state: state}).flatten()

    # Get policy
    def get_policy(self, state):
        return self.sess.run(self.policy, {self.state: state}).flatten()

    # Get policy and value
    def get_policy_and_value(self, state):
        policy, value = self.sess.run([self.policy, self.value], {self.state: state})
        return policy.flatten(), value.flatten()

    # Train the network on the given states and rewards
    def train(self, states, actions, target_values, advantages):
        self.sess.run(self.train_optimizer, feed_dict={self.state: states, self.action: actions,
                                                       self.target_value: target_values,
                                                       self.advantages: advantages})

    # Model for collecting visual information and calculating the policy vector and float value
    def build_model(self, h, w, channels):
        state = tf.placeholder(tf.float32, [None, w, h, channels], name='state')

        with tf.variable_scope('conv1'):
            conv1 = tf.contrib.layers.conv2d(inputs=state, num_outputs=16,
                                             kernel_size=[8, 8], stride=[4, 4], padding='VALID')
        with tf.variable_scope('conv2'):
            conv2 = tf.contrib.layers.conv2d(inputs=state, num_outputs=16,
                                             kernel_size=[4, 4], stride=[2, 2], padding='VALID')
        with tf.variable_scope('flatten'):
            flatten = tf.contrib.layers.flatten(conv2)

        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(flatten, num_outputs=256)
            
        with tf.variable_scope('policy'):
            policy = tf.contrib.layers.fully_connected(fc1, num_outputs=self.action_size,
                                                       activation_fn=tf.nn.softmax, biases_initializer=None)
        with tf.variable_scope('value'):
            value = tf.contrib.layers.fully_connected(fc1, num_outputs=1,
                                                      activation_fn=None, biases_initializer=None)
        return state, policy, value
