import paddle


def sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None):
    return paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=labels, ignore_index=- 100, name=name,
                                                                 normalize=False)
