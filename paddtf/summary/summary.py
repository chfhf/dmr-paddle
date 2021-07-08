import paddle
from visualdl import LogWriter

logwriter = LogWriter(logdir="./logs/")
def histogram(name, values, collections=None, family=None):
    # logwriter.add_histogram(tag=name,values=values,step=30)
    pass

def scalar(name, value):
    logwriter.add_scalar(name,value,step=1)