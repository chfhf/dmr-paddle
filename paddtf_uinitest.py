import paddtf as pf
import tensorflow as tf
import numpy as np

def test_Variable():
    value = 1.0
    name = "global_step"
    trainable=False
    pf_data=pf.varible_scope.Variable(name=name,value=value,trainable=trainable)
    tf_data=tf.Variable(name=name,value=value,trainable=trainable)
    print(pf_data,tf_data)
    # assert np.max(np.abs(np_out- torch_out))<=max_error, "Variable fail"



if __name__ == "__main__":
    test_Variable()