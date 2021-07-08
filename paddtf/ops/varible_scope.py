import paddle

def get_variable(name, shape,initializer=None, trainable=False):
    ret= paddle.fluid.layers.create_parameter(shape=shape, dtype='float32', name=name,default_initializer=initializer)
    if not trainable:
        ret.stop_gradient=True

    return ret


def Variable(value=0, name=None, trainable=False):
    ret= paddle.fluid.layers.create_parameter(shape=[1], dtype='float32', name='fc_b',default_initializer=paddle.nn.initializer.Constant(value))
    if not trainable:
        ret.stop_gradient=True

    return ret
                                           
def global_variables_initializer():
    print("global_variables_initializer not implemented")


def local_variables_initializer():
    print("local_variables_initializer not implemented")

