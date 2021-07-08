import paddle

def embedding_lookup(params,ids):
    ret=paddle.nn.functional.embedding(ids,params)
    return ret
    # oldshape=ids.shape+[params.shape[-1]]
    # ret= paddle.index_select(params,paddle.reshape(ids,shape=(-1,) ))
    #
    # return  paddle.reshape(ret,oldshape)
    # # return paddle.gather(x=params,index=ids)