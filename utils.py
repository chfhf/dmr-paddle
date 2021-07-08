import paddle.nn

import paddtf as tf
import numpy as np
from paddle.nn.functional import prelu



def deep_match(self,item_his_eb, context_his_eb, mask, match_mask, mid_his_batch, EMBEDDING_DIM, item_vectors, item_biases,
               n_mid):



    query = context_his_eb
    query = self.dm_align(query )
    query = self.dm_prelu(query )
    inputs = tf.concat([query, item_his_eb, query - item_his_eb, query * item_his_eb], axis=-1)  # B,T,E
    att_layer1 = tf.nn.sigmoid( self.dm_att_1(inputs ))
    att_layer2 = tf.nn.sigmoid( self.dm_att_2(att_layer1 ))
    att_layer3 = tf.nn.sigmoid( self.dm_att_3(att_layer2 ))
    scores = tf.transpose(att_layer3, [0, 2, 1])  # B,1,T

    # mask
    bool_mask = tf.equal(mask, tf.ones_like(mask))  # B,T
    key_masks = tf.expand_dims(bool_mask, 1)  # B,1,T
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)

    # tril
    scores_tile = tf.tile(tf.reduce_sum(scores, axis=1), [1, tf.shape(scores)[-1]])  # B, T*T
    scores_tile = tf.reshape(scores_tile, [-1, tf.shape(scores)[-1], tf.shape(scores)[-1]])  # B, T, T
    diag_vals = tf.ones_like(scores_tile)  # B, T, T
    # tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
    # TODO LinearOperatorLowerTriangular()

    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals)
    paddings = tf.ones_like(tril) * (-2 ** 32 + 1)
    scores_tile = tf.where( tril== 0, paddings, scores_tile)  # B, T, T
    scores_tile = tf.nn.softmax(scores_tile)  # B, T, T
    att_dm_item_his_eb = tf.matmul(scores_tile, item_his_eb)  # B, T, E


    dnn_layer1 = self.dm_fcn_1(att_dm_item_his_eb )


    dnn_layer1 = self.dm_fcn_1_prelu(dnn_layer1 )  # B, T, E

    # target mask
    user_vector = dnn_layer1[:, -1, :]

    user_vector2 = dnn_layer1[:, -2, :] * tf.reshape(match_mask, [-1, tf.shape(match_mask)[1], 1])[:, -2, :]
    num_sampled = 2000

    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=item_vectors,
                                                     biases=item_biases,
                                                     labels=tf.cast(tf.reshape(mid_his_batch[:, -1], [-1, 1]),
                                                                    tf.int64),
                                                     inputs=user_vector2,
                                                     num_sampled=num_sampled,
                                                     num_classes=n_mid,
                                                     sampled_values=tf.nn.learned_unigram_candidate_sampler(
                                                         tf.cast(tf.reshape(mid_his_batch[:, -1], [-1, 1]), tf.int64),
                                                         1, num_sampled, True, n_mid)
                                                     ))
    return loss, user_vector, scores


def dmr_fcn_attention(self,item_eb, item_his_eb, context_his_eb, mask, mode='SUM'):



    mask = tf.equal(mask, tf.ones_like(mask))
    item_eb_tile = tf.tile(item_eb, [1, tf.shape(mask)[1]])  # B, T*E
    item_eb_tile = tf.reshape(item_eb_tile, [-1, tf.shape(mask)[1], item_eb.shape[-1]])  # B, T, E
    if context_his_eb is None:
        query = item_eb_tile
    else:
        query = tf.concat([item_eb_tile, context_his_eb], axis=-1)
    query = self.dmr_align(query )
    query = self._dmr_prelu_0(query )
    dmr_all = tf.concat([query, item_his_eb, query - item_his_eb, query * item_his_eb], axis=-1)
    att_layer_1 = tf.nn.sigmoid(self.tg_att_1(dmr_all ))
    att_layer_2 = tf.nn.sigmoid(self.tg_att_2(att_layer_1 ))
    att_layer_3 =  self.tg_att_3(att_layer_2 )
    att_layer_3 = tf.reshape(att_layer_3, [-1, 1, tf.shape(item_his_eb)[1]])  # B,1,T
    scores = att_layer_3

    # Mask
    key_masks = tf.expand_dims(mask, 1)  # B,1,T
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    paddings_no_softmax = tf.zeros_like(scores)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

    scores = tf.nn.softmax(scores)

    if mode == 'SUM':
        output = tf.matmul(scores, item_his_eb)  # [B, 1, H]
        output = tf.reduce_sum(output, axis=1)  # B,E
    else:
        scores = tf.reshape(scores, [-1, tf.shape(item_his_eb)[1]])
        output = item_his_eb * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(item_his_eb))

    return output, scores, scores_no_softmax


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)

    pos, neg = 0.01, 0.01
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc



def calc_gauc(raw_arr_dict):
    gauc = 0.0
    cnt = 0
    for raw_arr in raw_arr_dict.values():
        if 1 not in np.array(raw_arr)[:, 1] or 0 not in np.array(raw_arr)[:, 1]:
            continue
        auc = calc_auc(raw_arr)
        gauc += auc * len(raw_arr)
        cnt += len(raw_arr)
    gauc = gauc / cnt
    return gauc
