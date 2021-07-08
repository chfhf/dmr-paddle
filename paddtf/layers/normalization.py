import paddle


def batch_normalization(inputs, momentum=0.99, epsilon=1e-3, name=None):
    return paddle.static.nn.batch_norm(input=inputs, momentum=momentum, epsilon=epsilon, param_attr=None,
                                       bias_attr=None, data_layout='NCHW', in_place=False, name=name)
