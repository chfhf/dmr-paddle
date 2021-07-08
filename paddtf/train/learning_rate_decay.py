import paddle


def exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False):
    return paddle.fluid.layers.exponential_decay(learning_rate, decay_steps, decay_rate, staircase=staircase)
