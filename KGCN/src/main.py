import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train
import tensorflow as tf
import csv

np.random.seed(555)


parser = argparse.ArgumentParser()

# movie
'''
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
'''

'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=3, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
'''


# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')



show_loss = False
show_time = False
show_topk = False

t = time()

args = parser.parse_args()
data = load_data(args)


for agg in ['sum']:#, 'concat', 'neighbor']:
    for k in range(4, 8):
        for dim in range(2, 8):
            for H in range(1, 5):
                file = open('results.csv', 'a', newline='', encoding='utf-8')
                writer = csv.writer(file)

                args = parser.parse_args()
                args.aggregator = agg
                args.neighbor_sample_size = 2 ** k
                args.dim = 2 ** dim
                args.n_iter = H
                data = load_data(args)
                auc, f1 = train(args, data, show_loss, show_topk)
                tf.reset_default_graph()
                writer.writerow(['music', agg, 2**k, 2**dim, H, auc, f1])
                print("finish with agg: {0}, k: {1}, dim: {2}, H: {3}".format(agg, 2 ** k, 2 ** dim, H))

                file.close()


if show_time:
    print('time used: %d s' % (time() - t))
