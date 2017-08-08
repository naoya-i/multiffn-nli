# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import argparse
from itertools import izip
import tensorflow as tf
import numpy as np

import utils
import ioutils

"""
Evaluate the performance of an NLI model on a dataset
"""


def print_errors(pairs, answers, label_dict):
    """
    Print the pairs for which the model gave a wrong answer,
    their gold label and the system one.
    """
    for pair, answer in izip(pairs, answers):
        label_str = pair[2]
        label_number = label_dict[label_str]
        if answer != label_number:
            sent1 = ' '.join(pair[0])
            sent2 = ' '.join(pair[1])
            print('Sent 1: {}\nSent 2: {}'.format(sent1, sent2))
            print('System label: {}, gold label: {}'.format(answer,
                                                            label_number))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
            
def print_insts(f, pairs, answers, logits, label_dict):
    """
    Print the pairs for which the model gave a wrong answer,
    their gold label and the system one.
    """
    label_dict_rev = dict(map(lambda (x, y): (y, x), (label_dict.iteritems())))
    
    for pair, answer, logit in izip(pairs, answers, logits):
        label_str = pair[2]
        label_number = label_dict[label_str]
        sent1 = ' '.join(pair[0])
        sent2 = ' '.join(pair[1])
        logit = softmax(logit)
        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            sent1, sent2,
            answer == label_number,
            label_dict_rev[answer], label_str,
            logit[answer]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Directory with saved model')
    parser.add_argument('dataset',
                        help='JSONL or TSV file with data to evaluate on')
    parser.add_argument('embeddings', help='Numpy embeddings file')
    parser.add_argument('vocabulary',
                        help='Text file with embeddings vocabulary')
    parser.add_argument('-v',
                        help='Verbose', action='store_true', dest='verbose')
    parser.add_argument('-e',
                        help='Print pairs and labels that got a wrong answer',
                        action='store_true', dest='errors')
    parser.add_argument('-i',
                        help='Print all instances to this file.',
                        dest='insts')
    args = parser.parse_args()

    utils.config_logger(verbose=args.verbose)
    params = ioutils.load_params(args.model)
    
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.InteractiveSession(config=session_config)

    model_class = utils.get_model_class(params)
    model = model_class.load(args.model, sess)
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings,
                                                    args.vocabulary,
                                                    generate=False,
                                                    load_extra_from=args.model,
                                                    normalize=True)
    model.initialize_embeddings(sess, embeddings)
    label_dict = ioutils.load_label_dict(args.model)

    pairs = ioutils.read_corpus(args.dataset, params['lowercase'],
                                params['language'])
    dataset = utils.create_dataset(pairs, word_dict, label_dict)
    loss, acc, answers, logits = model.evaluate(sess, dataset, True)
    print('# problems: %s' % dataset.num_items)
    print('Loss: %f' % loss)
    print('Accuracy: %f' % acc)

    if args.errors:
        print_errors(pairs, answers, label_dict)
        
    if None != args.insts:
        with open(args.insts, "w") as f:
            print_insts(f, pairs, answers, logits, label_dict)
