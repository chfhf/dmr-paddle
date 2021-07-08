import paddle

def softmax(x):
    return paddle.nn.functional.softmax(x, axis=- 1, dtype=None, name=None)


def sampled_softmax_loss(weights,biases,labels,inputs,num_sampled=0,num_classes=None,sampled_values=None):
    logits=paddle.matmul(inputs,weights,transpose_y=True)+paddle.reshape(biases,(1,-1))

    return paddle.fluid.layers.loss.sampled_softmax_with_cross_entropy(logits,labels, 
    num_samples=num_sampled, num_true=1, remove_accidental_hits=True, use_customized_samples=False,
     customized_samples=None, customized_probabilities=None, seed=0)
    # return paddle.nn.functional.softmax_with_cross_entropy(logits=logits,
    #                                                        label=labels,
    #                                                        soft_label=False,
    #                                                        ignore_index=- 100,
    #                                                        numeric_stable_mode=True,
    #                                                        return_softmax=False,
    #                                                        axis=- 1)
def learned_unigram_candidate_sampler(true_classes,num_true,num_sampled,unique,range_max,seed=None,name=None):

    return None

