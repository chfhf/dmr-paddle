import paddle


def dense(inputs, units, activation=None, name=None, ):
    return paddle.fluid.layers.fc(input=inputs, size=units, num_flatten_dims=1, act=activation, name=name)
