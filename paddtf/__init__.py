import paddle.nn.initializer
from . import linalg
from paddtf.framework import name_scope
from paddtf.ops import *
from paddtf.layers import *
from paddtf.framework import *
from paddtf.train import *
from paddtf.nn import *
from paddtf.client import *
from paddtf.summary import *
double = "float32"
bool = "bool"
float32 = "float32"
int32 = "int32"
int64 = "int64"

def zeros_initializer():
    return paddle.nn.initializer.Constant()