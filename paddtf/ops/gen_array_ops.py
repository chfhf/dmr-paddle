import paddle


def tile(input, multiples, name=None):
    return paddle.tile(input,multiples,name=name)
    # return paddle.expand(x=input, shape=multiples, name=name)


def reshape(tensor, shape, name=None):
    return paddle.reshape(x=tensor, shape=shape, name=name)
