import os
import argparse
import cPickle as pickle
from definitions import ROOT_PATH
from data import Data
from model import Model
from negations import assign_indices
import numpy as np


def get_dataset(max_len, validation_portion, batch_size):
    """
    Get a new preprocessed dataset. If an existing one is found, it is loaded instead.
    """
    prep_path = ROOT_PATH + '/data/prep_data_ml%d.pk' % max_len
    d = None

    if os.path.exists(prep_path):
        print 'Existing dataset found. Opening...'
        d = pickle.load(open(prep_path, 'rb'))

    if not d:
        d = Data(max_len)
        f = open(prep_path, 'wb')
        pickle.dump(d, f, 2)
        print '  New dataset saved.'

    # Initial shuffle and split
    d.get_train_values()
    d.shuffle()
    d.split_validation(validation_portion)

    t_rels = {}
    v_rels = {}
    print "Label frequencies:"
    for i in xrange(0, len(d.train_used)):
        r = d.train_used[i][2]
        if r in t_rels:
            t_rels[r] += 1
        else:
            t_rels[r] = 1
    for i in xrange(0, len(d.validation_used)):
        r = d.validation_used[i][2]
        if r in v_rels:
            v_rels[r] += 1
        else:
            v_rels[r] = 1
    for i in xrange(0, d.rel_count):
        if i in t_rels:
            t = t_rels[i]
        else:
            t = 0
        if i in v_rels:
            v = v_rels[i]
        else:
            v = 0
        print i, ": ", t, '~', v
    print 'train size: ', len(d.train)
    print 'test size: ', len(d.test)
    d.split_batches(batch_size)

    return d


def run(data, model, batch_size, epochs, early_stop, validation_portion, top_true, bottom_false, burn_in, name, check):
    train_batches = len(data.train_used)
    valid_batches = len(data.validation_used)
    test_batches = len(data.test_used)
    path = ROOT_PATH + '/saved/' + name
    if os.path.exists(path + '_weights.npz') and os.path.exists(path + '_stats.pk'):
        model.load(path)
    assign_indices(data)

    # Main loop
    for ep in xrange(model.last_epoch + 1, epochs+1):
        print 'Epoch %d...' % ep
        model.last_epoch = ep

        # 1: Train
        model.train(data, train_batches)

        # 2: Evaluate
        model.evaluate(data, valid_batches, True)

        # 3: Drop labels
        if top_true > 0 and ep >= burn_in:
            model.label_drop(data, valid_batches, top_true, bottom_false, check)

            # 4: Reshuffle and split off validation portion
            # Note: very unorthodox, technically cannot be called validation this way
            data.get_train_values()
            data.shuffle()
            data.split_validation(validation_portion)
            data.split_batches(batch_size)
            train_batches = len(data.train_used)
            valid_batches = len(data.validation_used)

        model.save(path)
        if model.epochs_since_best >= early_stop:
            print '  No more improvement seen after %d epochs, best was at %d' % (model.last_epoch, model.best_epoch)
            break

    # Test
    print 'TESTING'
    model.evaluate(data, test_batches, False)


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--epochs', metavar='epochs', nargs='?', type=int, default=25)
    parser.add_argument('--validation_portion', metavar='validation_portion', nargs='?', type=float, default=0.2,
                        help='Portion of the training data to be used for validation')
    parser.add_argument('--batch_size', metavar='batch_size', nargs='?', type=int, default=150)
    parser.add_argument('--max_len', metavar='max_len', nargs='?', type=int, default=100,
                        help='Maximum sentence length in the dataset')
    parser.add_argument('--learning_rate', metavar='learning_rate', nargs='?', type=float, default=0.00005)
    parser.add_argument('--units_lstm', metavar='units_sentence_embedder', nargs='?', type=int, default=70,
                        help='Number of hidden units in the LSTM / filters in the CNN')
    parser.add_argument('--word_emb_size', metavar='word_emb_size', nargs='?', type=int, default=30)
    parser.add_argument('--early_stop', metavar='early_stop', nargs='?', type=int, default=100,
                        help='Number of epochs without improvement before the model stops')
    parser.add_argument('--dropout_probability', metavar='dropout_probability', nargs='?', type=int, default=0.5)
    parser.add_argument('--top_true', metavar='top_true', nargs='?', type=int, default=3,
                        help='Number of top labels in each evaluation to assume as true. Set to <1 to ignore')
    parser.add_argument('--bottom_false', metavar='bottom_false', nargs='?', type=int, default=10,
                        help='Number of bottom labels in which labels become a drop candidate')
    parser.add_argument('--check', metavar='check', nargs='?', type=bool, default=True,
                        help='Check for negation labels before dropping')
    parser.add_argument('--burn_in', metavar='burn_in', nargs='?', type=int, default=5,
                        help='Number of epochs before dropping')
    parser.add_argument('--name', metavar='name', nargs='?', type=str, default='drop_check',
                        help='Name of this run')
#    parser.add_argument('')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    data = get_dataset(args.max_len, args.validation_portion, args.batch_size)

    model = Model(data, args.batch_size, args.learning_rate, args.units_lstm, args.word_emb_size,
                  args.dropout_probability)

    run(data, model, args.batch_size, args.epochs, args.early_stop, args.validation_portion, args.top_true,
       args.bottom_false, args.burn_in, args.name, args.check)
