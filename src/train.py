# -*- coding: utf-8 -*-

from __future__ import division, print_function

"""
Script to train an RTE LSTM.
"""

import os
import sys
import argparse
import tensorflow as tf

import ioutils
import utils
from classifiers import LSTMClassifier, MultiFeedForwardClassifier,\
    DecomposableNLIModel

import numpy as np

np.random.seed(19851008)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('embeddings',
                        help='Text or numpy file with word embeddings')
    parser.add_argument('train', help='PICKLE, JSONL or TSV file with training corpus')
    parser.add_argument('validation',
                        help='PICKLE, JSONL or TSV file with validation corpus')
    parser.add_argument('save', help='Directory to save the model files')
    parser.add_argument('model', help='Type of architecture',
                        choices=['lstm', 'mlp'])
    parser.add_argument('--vocab', help='Vocabulary file (only needed if numpy'
                                        'embedding file is given)')
    parser.add_argument('-e', dest='num_epochs', default=10, type=int,
                        help='Number of epochs')
    parser.add_argument('-b', dest='batch_size', default=32, help='Batch size',
                        type=int)
    parser.add_argument('-u', dest='num_units', help='Number of hidden units',
                        default=100, type=int)
    parser.add_argument('--no-proj', help='Do not project input embeddings to '
                                          'the same dimensionality used by '
                                          'internal networks',
                        action='store_false', dest='no_project')
    parser.add_argument('-d', dest='dropout', help='Dropout keep probability',
                        default=1.0, type=float)
    parser.add_argument('-c', dest='clip_norm', help='Norm to clip training '
                                                     'gradients',
                        default=100, type=float)
    parser.add_argument('-r', help='Learning rate', type=float, default=0.001,
                        dest='rate')
    parser.add_argument('--lang', choices=['en', 'pt'], default='en',
                        help='Language (default en; only affects tokenizer)')
    parser.add_argument('--lower', help='Lowercase the corpus (use it if the '
                                        'embedding model is lowercased)',
                        action='store_true')
    parser.add_argument('--use-intra', help='Use intra-sentence attention',
                        action='store_true', dest='use_intra')
    parser.add_argument('--l2', help='L2 normalization constant', type=float,
                        default=0.0)
    parser.add_argument('--report', help='Number of batches between '
                                         'performance reports',
                        default=100, type=int)
    parser.add_argument('-v', help='Verbose', action='store_true',
                        dest='verbose')
    parser.add_argument('--optim', help='Optimizer algorithm',
                        default='adagrad',
                        choices=['adagrad', 'adadelta', 'adam'])
    parser.add_argument('-a', dest='additional_training',
                        help='Additional training corpus (PICKLE, JSONL or TSV)')
    parser.add_argument('--shuffle-by-bucket', help='Shuffle the training data by bucket',
                        action='store_true', dest='shuffle_by_bucket')
    parser.add_argument('--report-after', help='Frequently report after this epoch.',
                        default=0, type=int)
    parser.add_argument('--continue', help='Continue training.',
                        action='store_true', dest='cont')
    parser.add_argument('--warm-start', help='Use pre-trained model.',
                        dest='warm')
    
    args = parser.parse_args()

    utils.config_logger(args.verbose)
    logger = utils.get_logger('train')
    logger.debug('Training with following options: %s' % ' '.join(sys.argv))
    train_pairs = ioutils.read_corpus(args.train, args.lower, args.lang)
    valid_pairs = ioutils.read_corpus(args.validation, args.lower, args.lang)

    if args.additional_training != None:
        train_pairs += ioutils.read_corpus(args.additional_training, args.lower, args.lang)

    assert(not args.cont) # Not implemented yet.
    
    # whether to generate embeddings for unknown, padding, null
    is_really_cont = args.warm != None or (args.cont and os.path.exists(os.path.join(args.save, "model.meta")))
    warmup_model = args.warm
        
    if is_really_cont:
        logger.info('Found a model. Fine-tuning...')
        
        word_dict, embeddings = ioutils.load_embeddings(args.embeddings, args.vocab,
                                                        generate=False, normalize=True,
                                                        load_extra_from=warmup_model)
        params = ioutils.load_params(warmup_model)
        
    else:        
        word_dict, embeddings = ioutils.load_embeddings(args.embeddings, args.vocab,
                                                        generate=True, normalize=True)
        ioutils.write_params(args.save, lowercase=args.lower, language=args.lang,
                             model=args.model)
        ioutils.write_extra_embeddings(embeddings, args.save)
        
    logger.info('Converting words to indices')
    # find out which labels are there in the data
    # (more flexible to different datasets)
    label_dict = utils.create_label_dict(train_pairs)
    train_data = utils.create_dataset(train_pairs, word_dict, label_dict)
    valid_data = utils.create_dataset(valid_pairs, word_dict, label_dict)

    ioutils.write_label_dict(label_dict, args.save)
    
    logger.info('{} items in training data.'.format(train_data.num_items))

    msg = '{} sentences have shape {} (firsts) and {} (seconds)'
    logger.debug(msg.format('Training',
                            train_data.sentences1.shape,
                            train_data.sentences2.shape))
    logger.debug(msg.format('Validation',
                            valid_data.sentences1.shape,
                            valid_data.sentences2.shape))

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.InteractiveSession(config=session_config)
    logger.info('Creating model')
    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    if is_really_cont:
        model_class = utils.get_model_class(params)
        model, saver = model_class.load(warmup_model, sess, training=True, embeddings=embeddings)

    else:
        if args.model == 'mlp':
            model = MultiFeedForwardClassifier(args.num_units, 3, vocab_size,
                                               embedding_size,
                                               use_intra_attention=args.use_intra,
                                               training=True,
                                               project_input=args.no_project,
                                               optimizer=args.optim)
        else:
            model = LSTMClassifier(args.num_units, 3, vocab_size,
                                   embedding_size, training=True,
                                   project_input=args.no_project,
                                   optimizer=args.optim)

        model.initialize(sess, embeddings)

    # this assertion is just for type hinting for the IDE
    assert isinstance(model, DecomposableNLIModel)

    total_params = utils.count_parameters()
    logger.debug('Total parameters: %d' % total_params)

    logger.info('Starting training')
    model.train(sess, train_data, valid_data, args.save, args.rate,
                args.num_epochs, args.batch_size, args.dropout, args.l2,
                args.clip_norm, args.report, args.shuffle_by_bucket,
                args.report_after, saver if is_really_cont else None,
    )
