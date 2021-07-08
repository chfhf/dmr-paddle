# coding: utf-8
import os

import paddle

import argparse
# from model_dygraph import *
from model_baseline import *
from data_iterator_paddle import *
import numpy as np
import paddtf as tf
import sys
import random
from datetime import timedelta, datetime

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', help='num epochs',type=int, default=10)
parser.add_argument('--batch_size', help='batch_size', type=int, default=2560)
parser.add_argument('--window_size', help='window_size', type=int, default=50)
parser.add_argument('--starter_learning_rate', help='starter_learning_rate', type=float, default=0.001)
parser.add_argument('--learning_rate_decay', help='learning_rate_decay', type=int, default=1.0)
parser.add_argument('--data_path', help='data_path', type=str, default="/home/aistudio/work/")
parser.add_argument('--ckpt_dir', help='ckpt_dir', type=str, default='ckpt/baseline_')
parser.add_argument('--load_model', help='model checkpoint path', type=str, default='')
parser.add_argument('--mode', help='mode', choices=['train', 'test', 'demo'],type=str, default='train')
args = parser.parse_args()


num_epochs = args.num_epochs
batch_size =  args.batch_size
window_size = args.window_size
starter_learning_rate = args.starter_learning_rate
learning_rate_decay = args.learning_rate_decay
data_path= args.data_path
today = datetime.today() + timedelta(0)
ckpt_dir = args.ckpt_dir + f'{os.path.basename(data_path)}_epoch{num_epochs}_bs{batch_size}_lr{starter_learning_rate}'
os.makedirs(ckpt_dir,exist_ok=True)
load_model=args.load_model

in_features = 16
def full_train():
    lr_scheduler=paddle.optimizer.lr.ExponentialDecay(starter_learning_rate,gamma=learning_rate_decay,last_epoch=2000000)
    global load_model
    # construct the model structure
    model = Model_DMR(in_features)
    adam_optim = paddle.optimizer.Adam(learning_rate=lr_scheduler,parameters=model.parameters())
    if len(load_model)>1:
        model.load_dict(paddle.load(load_model))
        print("successfully loaded model from ",load_model)
    import glob
    train_data=None
    train_fns=glob.glob(data_path+"/alimama_train_*.txt.gz")
    if data_path.endswith(".csv"):
        train_data = paddle.io.DataLoader.from_generator(capacity=1 )
        train_data.set_batch_generator(reader_data(data_path, batch_size, 20))
        train_fns = [data_path]
        print("loaded training data file:",data_path)
    best_loss=10000000000
    for epoch in range(num_epochs):
        for train_fn in train_fns:
            if not data_path.endswith(".csv"):
                train_data = paddle.io.DataLoader.from_generator(capacity=1,use_multiprocess=True,use_double_buffer=True)
                train_data.set_batch_generator(reader_data(train_fn, batch_size, 20))  
                print("loaded training data file:",train_fn)
            iter = 0
            test_iter = 100
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            stored_arr = []
            for features, targets in train_data:
                loss, acc, aux_loss, prob = model.train_batch(None, features, targets)
                loss.backward()
                adam_optim.minimize(loss)
                adam_optim.clear_grad()
                loss_sum += loss.numpy()
                accuracy_sum += acc.numpy()
                aux_loss_sum += aux_loss.numpy()
                prob_1 = prob.numpy()[:, 0].tolist()
                target_1 = targets.numpy().tolist()
                for p, t in zip(prob_1, target_1):
                    stored_arr.append([p, t])
                iter += 1
                if (iter % test_iter) == 0:
                    print(datetime.now().ctime())
                    print(
                        '%s|EPOCH:%d |iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f ---- train_auc: %.4f' % \
                        (os.path.basename(data_path).replace(".csv",""),epoch,iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter,
                        calc_auc(stored_arr)))
                    if best_loss>loss_sum:
                        best_loss=loss_sum
                        paddle.save(model.state_dict(), ckpt_dir + "/best" )
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                    stored_arr = []
                    paddle.save(model.state_dict(), ckpt_dir + "/current" )

    print("session finished.")

from csvdataset import  CSVDataset
def small_train():
    print("DEMO Purpose only: training on small sample dataset")
    batch_size=256
    # train_data = paddle.io.DataLoader.from_generator(capacity=1,use_multiprocess=True,use_double_buffer=True)
    # train_data.set_batch_generator(reader_data('alimama_sampled.txt', batch_size, 20))
    # train_data = reader_data('alimama_sampled.txt', batch_size, 20)()

    train_data=CSVDataset('alimama_sampled.txt')
    train_data=paddle.io.DataLoader(train_data,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=False)

    lr_scheduler=paddle.optimizer.lr.ExponentialDecay(starter_learning_rate,gamma=learning_rate_decay,last_epoch=2000000)

    # construct the model structure
    model = Model_DMR(in_features)
    adam_optim = paddle.optimizer.Adam(learning_rate=lr_scheduler,parameters=model.parameters())
    for epoch in range(num_epochs):
        iter = 0
        test_iter = 10
        loss_sum = 0.0
        accuracy_sum = 0.
        aux_loss_sum = 0.
        stored_arr = []


        for features, targets in train_data:
            loss, acc, aux_loss, prob = model.train_batch(None, features, targets)
           
            loss.backward()
            adam_optim.minimize(loss)
            adam_optim.clear_grad()
            loss_sum += loss.numpy()
            accuracy_sum += acc.numpy()
            aux_loss_sum += aux_loss.numpy()
            prob_1 = prob.numpy()[:, 0].tolist()
            target_1 = targets.numpy().tolist()
            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, t])
            iter += 1
            if (iter % test_iter) == 0:
                print(datetime.now().ctime())
                auc_score=calc_auc(stored_arr)
                tf.summary.scalar("train/auc",auc_score)
                print(
                    'EPOCH:%d |iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f ---- train_auc: %.4f' % \
                    (epoch,iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter,
                      auc_score ))
                loss_sum = 0.0
                accuracy_sum = 0.0
                aux_loss_sum = 0.0
                stored_arr = []
                # paddle.save(model.state_dict(),ckpt_dir+"/iter_"+str(iter)) ##see if model size very big
                paddle.save(model.state_dict(), ckpt_dir + "/current" )

    print("session finished.")

def eval():
    print("Evaluate On testing data")
    test_data = paddle.io.DataLoader.from_generator(capacity=1 )
    test_data.set_batch_generator(reader_data(data_path+"alimama_test.txt.gz", batch_size, 20))
    model = Model_DMR(in_features)
    iter = 0
    test_iter = 100
    loss_sum = 0.0
    accuracy_sum = 0.
    aux_loss_sum = 0.
    stored_arr = []
    global  load_model
    if len(load_model) > 1:
        model.load_dict(paddle.load(load_model))
        print("successfully loaded model from ", load_model)

    for features, targets in test_data:
        loss, acc, aux_loss, prob = model.calculate(None, features, targets)
        loss_sum += loss.numpy()
        accuracy_sum += acc.numpy()
        aux_loss_sum += aux_loss.numpy()
        prob_1 = prob[:, 0].numpy().tolist()
        target_1 = targets.numpy().tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
        iter += 1
        if (iter % test_iter) == 0:
            print(datetime.now().ctime())
            auc_score=calc_auc(stored_arr)
            tf.summary.scalar("test/auc",auc_score)
            print(
                'iter: %d ----> test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- test_auc: %.4f' % \
                (iter, loss_sum / iter, accuracy_sum / iter, aux_loss_sum / iter, auc_score))
    print("session finished.")


if __name__ == "__main__":
    SEED = 3
    #
    print("Parameters:",args)
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if args.mode == 'train':
        full_train()
    elif args.mode == 'test':
        eval()
    else:
        small_train()
