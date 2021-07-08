import paddle


def placeholder(dtype, shape, name, lod_level=0):
    return paddle.static.data(dtype=dtype, shape=shape, name=name, lod_level=lod_level)


def expand_dims(input, axis=None, name=None, dim=None):
    return paddle.fluid.layers.unsqueeze(input, axes=axis, name=name)


def concat(values, axis, name="concat"):
    return paddle.concat(x=values, axis=axis)


def shape(input):
    return paddle.shape(input)


def transpose(a, perm=None, name="transpose"):
    return paddle.transpose(x=a, perm=perm, name=name)


def ones_like(tensor, dtype=None, name=None):
    return paddle.ones_like(x=tensor, dtype=dtype, name=name)

def zeros_like(tensor, dtype=None, name=None):
    return paddle.zeros_like(x=tensor, dtype=dtype, name=name)

def where(condition, x=None, y=None, name=None):
    return paddle.where(condition, x, y, name=name)
