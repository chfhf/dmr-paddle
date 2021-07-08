# coding: utf-8
import paddle.nn

from utils import *

# user feature size
user_size = 1141730
cms_segid_size = 97
cms_group_id_size = 13
final_gender_code_size = 3
age_level_size = 7
pvalue_level_size = 4
shopping_level_size = 4
occupation_size = 3
new_user_class_level_size = 5

# item feature size
adgroup_id_size = 846812
cate_size = 12978
campaign_id_size = 423437
customer_size = 255876
brand_size = 461529

# context feature size
btag_size = 5
pid_size = 2

# embedding size
main_embedding_size = 32
other_embedding_size = 8


class Model(paddle.nn.Layer):
    def __init__(self,  in_features):
        super(Model, self).__init__()
        self.aux_loss = paddle.zeros([1])
        # input
        self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [user_size, main_embedding_size])
        self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [adgroup_id_size, main_embedding_size])
        self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [cate_size, main_embedding_size])
        self.brand_embeddings_var = tf.get_variable("brand_embedding_var", [brand_size, main_embedding_size])
        self.btag_embeddings_var = tf.get_variable("btag_embedding_var", [btag_size, other_embedding_size])
        self.dm_btag_embeddings_var = tf.get_variable("dm_btag_embedding_var", [btag_size, other_embedding_size])

        self.campaign_id_embeddings_var = tf.get_variable("campaign_id_embedding_var",
                                                          [campaign_id_size, main_embedding_size])

        self.customer_embeddings_var = tf.get_variable("customer_embedding_var",
                                                       [customer_size, main_embedding_size])

        self.cms_segid_embeddings_var = tf.get_variable("cms_segid_embedding_var",
                                                        [cms_segid_size, other_embedding_size])

        self.cms_group_id_embeddings_var = tf.get_variable("cms_group_id_embedding_var",
                                                           [cms_group_id_size, other_embedding_size])
        self.final_gender_code_embeddings_var = tf.get_variable("final_gender_code_embedding_var",
                                                                    [final_gender_code_size, other_embedding_size])

        self.age_level_embeddings_var = tf.get_variable("age_level_embedding_var",
                                                        [age_level_size, other_embedding_size])

        self.pvalue_level_embeddings_var = tf.get_variable("pvalue_level_embedding_var",
                                                           [pvalue_level_size, other_embedding_size])

        self.shopping_level_embeddings_var = tf.get_variable("shopping_level_embedding_var",
                                                             [shopping_level_size, other_embedding_size])

        self.occupation_embeddings_var = tf.get_variable("occupation_embedding_var",
                                                         [occupation_size, other_embedding_size])

        self.new_user_class_level_embeddings_var = tf.get_variable("new_user_class_level_embedding_var",
                                                                   [new_user_class_level_size,
                                                                    other_embedding_size])

        self.pid_embeddings_var = tf.get_variable("pid_embedding_var", [pid_size, other_embedding_size])


        self.bn_inp=paddle.nn.BatchNorm(num_channels=393)
        self.f0=paddle.nn.Linear(in_features=393,out_features=1)
        # self.f1 = paddle.nn.Linear(in_features=512, out_features=256)
        # self.prelu0=paddle.nn.PReLU()
        # self.prelu1 = paddle.nn.PReLU()
        # self.f2= paddle.nn.Linear(in_features=256, out_features=128)
        # self.prelu2 = paddle.nn.PReLU()
        # self.f3 = paddle.nn.Linear(in_features=128, out_features=1)

        # out_features=64
        # self.dm_align=paddle.nn.Linear(in_features=in_features,out_features=out_features)
        # self.dm_att_1=paddle.nn.Linear(in_features=256,out_features=80)
        # self.dm_att_2=paddle.nn.Linear(in_features=80,out_features=40)
        # self.dm_att_3 = paddle.nn.Linear(in_features=40, out_features=1)
        # EMBEDDING_DIM =main_embedding_size
        # self.dm_fcn_1 = paddle.nn.Linear(in_features=64, out_features=EMBEDDING_DIM)
        # self.dm_prelu=paddle.nn.PReLU()
        # self.dm_fcn_1_prelu = paddle.nn.PReLU()
        #
        # query_shape_lastdim=80
        # out_features2=64
        # dmr_all_lastdim=256
        # self.dmr_align=paddle.nn.Linear(query_shape_lastdim,out_features2)
        # self._dmr_prelu_0=paddle.nn.PReLU()
        # self.tg_att_1=paddle.nn.Linear(dmr_all_lastdim,80)
        # self.tg_att_2 = paddle.nn.Linear(80, 40)
        # self.tg_att_3 = paddle.nn.Linear(40, 1)
        print("finished model initialization")

    def forward(self, feature_ph, target_ph,tag=""):

        self.feature_ph = feature_ph
        self.target_ph = target_ph

        ## behavior history feature
        self.btag_his = tf.cast(self.feature_ph[:, 0:50], tf.int32)
        self.cate_his = tf.cast(self.feature_ph[:, 50:100], tf.int32)
        self.brand_his = tf.cast(self.feature_ph[:, 100:150], tf.int32)
        self.mask = tf.cast(self.feature_ph[:, 150:200], tf.int32)
        self.match_mask = tf.cast(self.feature_ph[:, 200:250], tf.int32)

        # user side features
        self.uid = tf.cast(self.feature_ph[:, 250], tf.int32)
        self.cms_segid = tf.cast(self.feature_ph[:, 251], tf.int32)
        self.cms_group_id = tf.cast(self.feature_ph[:, 252], tf.int32)
        self.final_gender_code = tf.cast(self.feature_ph[:, 253], tf.int32)
        self.age_level = tf.cast(self.feature_ph[:, 254], tf.int32)
        self.pvalue_level = tf.cast(self.feature_ph[:, 255], tf.int32)
        self.shopping_level = tf.cast(self.feature_ph[:, 256], tf.int32)
        self.occupation = tf.cast(self.feature_ph[:, 257], tf.int32)
        self.new_user_class_level = tf.cast(self.feature_ph[:, 258], tf.int32)

        ##ad side features
        self.mid = tf.cast(self.feature_ph[:, 259], tf.int32)
        self.cate_id = tf.cast(self.feature_ph[:, 260], tf.int32)
        self.campaign_id = tf.cast(self.feature_ph[:, 261], tf.int32)
        self.customer = tf.cast(self.feature_ph[:, 262], tf.int32)
        self.brand = tf.cast(self.feature_ph[:, 263], tf.int32)
        self.price = tf.expand_dims(tf.cast(self.feature_ph[:, 264], tf.float32)*(1e-6), 1)

        self.pid = tf.cast(self.feature_ph[:, 265], tf.int32)



        tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
        self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid)


        tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
        self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid)


        tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
        self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cate_id)
        self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cate_his)


        self.brand_batch_embedded = tf.nn.embedding_lookup(self.brand_embeddings_var, self.brand)
        self.brand_his_batch_embedded = tf.nn.embedding_lookup(self.brand_embeddings_var, self.brand_his)


        self.btag_his_batch_embedded = tf.nn.embedding_lookup(self.btag_embeddings_var, self.btag_his)

        self.dm_btag_his_batch_embedded = tf.nn.embedding_lookup(self.dm_btag_embeddings_var, self.btag_his)


        self.campaign_id_batch_embedded = tf.nn.embedding_lookup(self.campaign_id_embeddings_var, self.campaign_id)

        self.customer_batch_embedded = tf.nn.embedding_lookup(self.customer_embeddings_var, self.customer)


        self.cms_segid_batch_embedded = tf.nn.embedding_lookup(self.cms_segid_embeddings_var, self.cms_segid)


        self.cms_group_id_batch_embedded = tf.nn.embedding_lookup(self.cms_group_id_embeddings_var,
                                                                  self.cms_group_id)


        self.final_gender_code_batch_embedded = tf.nn.embedding_lookup(self.final_gender_code_embeddings_var,
                                                                       self.final_gender_code)


        self.age_level_batch_embedded = tf.nn.embedding_lookup(self.age_level_embeddings_var, self.age_level)


        self.pvalue_level_batch_embedded = tf.nn.embedding_lookup(self.pvalue_level_embeddings_var,
                                                                  self.pvalue_level)


        self.shopping_level_batch_embedded = tf.nn.embedding_lookup(self.shopping_level_embeddings_var,
                                                                    self.shopping_level)


        self.occupation_batch_embedded = tf.nn.embedding_lookup(self.occupation_embeddings_var, self.occupation)


        self.new_user_class_level_batch_embedded = tf.nn.embedding_lookup(self.new_user_class_level_embeddings_var,
                                                                          self.new_user_class_level)


        self.pid_batch_embedded = tf.nn.embedding_lookup(self.pid_embeddings_var, self.pid)

        self.user_feat = tf.concat(
            [self.uid_batch_embedded, self.cms_segid_batch_embedded, self.cms_group_id_batch_embedded,
             self.final_gender_code_batch_embedded, self.age_level_batch_embedded, self.pvalue_level_batch_embedded,
             self.shopping_level_batch_embedded, self.occupation_batch_embedded,
             self.new_user_class_level_batch_embedded], -1)
        self.item_his_eb = tf.concat([self.cat_his_batch_embedded, self.brand_his_batch_embedded], -1)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        self.item_feat = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded, self.brand_batch_embedded,
                                    self.campaign_id_batch_embedded, self.customer_batch_embedded, self.price], -1)
        self.item_eb = tf.concat([self.cat_batch_embedded, self.brand_batch_embedded], -1)
        self.context_feat = self.pid_batch_embedded




    def build_fcn_net(self, inp,tag=""):


        # inp = self.bn_inp(inp)
        dnn3 = self.f0(inp )
        # dnn0 = self.prelu0(dnn0 )
        # dnn1 = self.f1(dnn0 )
        # dnn1 = self.prelu0(dnn1 )
        # dnn2 = self.f2(dnn1 )
        # dnn2 = self.prelu2(dnn2 )
        # dnn3 = self.f3(dnn2 )
        self.y_hat = tf.nn.sigmoid(dnn3)

        with tf.name_scope('Metrics'):
            if self.target_ph is not None:
                # Cross-entropy loss and optimizer initialization
                # print("mean self.target_ph:",paddle.mean(self.target_ph.astype("float32")))
                positive_rate=tf.reduce_mean(self.target_ph.astype("float32"))
                ##rebalance the class weight
                weights=paddle.index_select(tf.concat([positive_rate,1-positive_rate],0), self.target_ph.astype("int"))
                # print(weights.shape)
                ctr_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_ph, logits=tf.reduce_sum(dnn3, 1)*weights))
                

                ####explict optimize AUC based on https://github.com/mlarionov/machine_learning_POC/blob/master/auc/pairwise.ipynb
                # activations=paddle.reshape(tf.reduce_sum(dnn3, 1),(-1,1))
                # y=paddle.reshape(self.target_ph.astype("float32") ,(-1,1))
                # part1=tf.sigmoid(activations @ tf.transpose(activations,[1,0]))
                # ones_y=paddle.ones(y.shape)
                # part2=paddle.maximum(y @ tf.transpose(ones_y,[1,0]) - ones_y@ tf.transpose(y,[1,0]),paddle.zeros((y.shape[0],y.shape[0]) ))
                # ctr_loss = - tf.reduce_mean( part1* part2)*10

                self.ctr_loss = ctr_loss
             
                self.loss = ctr_loss + self.aux_loss
                tf.summary.scalar(tag+'loss', self.loss)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                tf.summary.scalar(tag+'accuracy', self.accuracy)
                return self.y_hat ,self.loss,self.accuracy,self.aux_loss
            else:
                return self.y_hat ,None,None,None


    def train_batch(self, sess, features, targets):
        self.train()
        self.y_hat ,self.loss,self.accuracy,self.aux_loss=self.forward(  features,  targets,tag="train/")
        loss, accuracy, aux_loss, probs =self.loss, self.accuracy, self.aux_loss,  self.y_hat
        return loss, accuracy, aux_loss, probs

    def calculate(self, sess, features, targets):
        self.eval()
        self.y_hat ,self.loss,self.accuracy,self.aux_loss=self.forward( paddle.to_tensor(features), paddle.to_tensor(targets),tag="test/")
        loss, accuracy, aux_loss, probs =self.loss, self.accuracy, self.aux_loss,  self.y_hat
        return loss, accuracy, aux_loss, probs


class Model_DMR(Model):
    def __init__(self, in_features):
        super(Model_DMR, self).__init__(in_features)

        # self.position_his = tf.range(50)
        # self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, other_embedding_size])
        # self.dm_position_embeddings_var = tf.get_variable("dm_position_embeddings_var", [50, other_embedding_size])
        # self.dm_item_vectors = tf.get_variable("dm_item_vectors", [cate_size, main_embedding_size])
        # self.dm_item_biases = tf.get_variable('dm_item_biases', [cate_size], initializer=tf.zeros_initializer(),
        #                                      trainable=False)
        # self.dm_position_his = tf.range(50)

    def forward(self, feature_ph, target_ph,tag=""):
        super(Model_DMR,self).forward(feature_ph, target_ph,tag)

        # self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        # self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.mid)[0], 1])  # B*T,E
        # self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.mid)[0], -1,
        #                                                          self.position_his_eb.shape[
        #                                                              1]])  # B,T,E
        #
        #
        #
        # self.dm_position_his_eb = tf.nn.embedding_lookup(self.dm_position_embeddings_var, self.dm_position_his)  # T,E
        # self.dm_position_his_eb = tf.tile(self.dm_position_his_eb, [tf.shape(self.mid)[0], 1])  # B*T,E
        # self.dm_position_his_eb = tf.reshape(self.dm_position_his_eb, [tf.shape(self.mid)[0], -1,
        #                                                                self.dm_position_his_eb.shape[
        #                                                                    1]])  # B,T,E
        #
        # self.position_his_eb = tf.concat([self.position_his_eb, self.btag_his_batch_embedded], -1)
        # self.dm_position_his_eb = tf.concat([self.dm_position_his_eb, self.dm_btag_his_batch_embedded], -1)

        # # User-to-Item Network
        #
        # with tf.name_scope('u2i_net'):
        #
        #     tf.summary.histogram('dm_item_vectors', self.dm_item_vectors)
        #
        #     # Auxiliary Match Network
        #     self.aux_loss, dm_user_vector, scores = deep_match(self,self.item_his_eb, self.dm_position_his_eb, self.mask,
        #                                                        tf.cast(self.match_mask, tf.float32), self.cate_his,
        #                                                        main_embedding_size, self.dm_item_vectors, self.dm_item_biases,
        #                                                        cate_size)
        #     self.aux_loss *= 0.1
        #     dm_item_vec = tf.nn.embedding_lookup(self.dm_item_vectors, self.cate_id)  # B,E
        #     rel_u2i = tf.reduce_sum(dm_user_vector * dm_item_vec, axis=-1, keep_dims=True)  # B,1
        #     self.rel_u2i = rel_u2i
        #
        # # Item-to-Item Network
        # with tf.name_scope('i2i_net'):
        #     att_outputs, alphas, scores_unnorm = dmr_fcn_attention(self,self.item_eb, self.item_his_eb, self.position_his_eb,
        #                                                            self.mask)
        #     tf.summary.histogram('att_outputs', alphas)
        #     rel_i2i = tf.expand_dims(tf.reduce_sum(scores_unnorm, [1, 2]), -1)
        #     self.rel_i2i = rel_i2i
        #     self.scores = tf.reduce_sum(alphas, 1)

        inp = tf.concat([self.user_feat, self.item_feat, self.context_feat, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum ], -1)
        return self.build_fcn_net(inp,tag)
