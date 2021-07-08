import paddle
import numpy as np


def cast(x, dtype, name=None):
    return paddle.cast(x, dtype)


def reduce_sum(input_tensor, axis=None, keep_dims=None):
    return paddle.fluid.layers.reduce_sum(input=input_tensor, dim=axis, keep_dim=keep_dims, name=None)


def sigmoid(x, name=None):
    return paddle.nn.functional.sigmoid(x, name=name)


def reduce_mean(input_tensor, axis=None, keepdims=None, name=None):
    return paddle.fluid.layers.reduce_mean(input=input_tensor, dim=axis, keep_dim=keepdims, name=name)


def round(x, name=None):
    return paddle.round(x, name)


def equal(x, y, name=None):
    return paddle.equal(x, y, name=name)


def range(start, limit=None, delta=1, dtype=None, name="range"):
    return paddle.arange(start=0,end=start)


def matmul(a, b, transpose_a=False, transpose_b=False, name=None):
    return paddle.matmul(x=a, y=b, transpose_x=transpose_a, transpose_y=transpose_b, name=name)
