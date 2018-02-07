"""
Summary
"""
import tensorflow as tf

# Get status of episode average reward and average value
class Summary():
    def __init__(self, logdir, agent):
        with tf.variable_scope('summary'):
            summarising = ['episode_avg_reward', 'avg_value']
            self.agent = agent
            self.writer = tf.summary.FileWriter(logdir, self.agent.sess.graph)
            self.summary_ops, self.summary_vars, self.summary_ph = {}, {}, {}
            for s in summarising:
                self.summary_vars[s] = tf.Variable(0.0)
                self.summary_ops[s] = tf.summary.scalar(s, self.summary_vars[s])
                self.summary_ph[s] = tf.placeholder(tf.float32, name=s)
            self.update_ops = []
            for k in self.summary_vars:
                self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
            self.summary_op = tf.summary.merge(list(self.summary_ops.values()))

    def write_summary(self, summary, t):
        self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in summary.items()})
        summary_to_add = self.agent.sess.run(self.summary_op, {self.summary_vars[k]: v for k, v in summary.items()})
        self.writer.add_summary(summary_to_add, global_step=t)
