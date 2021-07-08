import paddle
def LinearOperatorLowerTriangular( input):
    return paddle.tensor.tril(input)