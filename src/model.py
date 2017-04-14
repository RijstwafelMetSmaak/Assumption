import numpy as np
import lasagne
import theano
import theano.tensor as T
import cPickle as pickle
from negations import negations_for_label
from sklearn import metrics
from definitions import NEGATIVE_LABEL


class ModelStats:
    """
    Lightweight class containing the Model's results. Used for pickle dump.
    """
    def __init__(self, last_epoch, best_epoch, best_pr_area, epochs_since_best,
                 precision_list, recall_list, f1_measure_list, pr_area_list, drops):
        self.last_epoch = last_epoch
        self.best_epoch = best_epoch
        self.best_pr_area = best_pr_area
        self.epochs_since_best = epochs_since_best
        self.precision_list = precision_list
        self.recall_list = recall_list
        self.f1_measure_list = f1_measure_list
        self.pr_area_list = pr_area_list
        self.drops = drops


class Model:
    """
    Main class to wrap the model together.
    """

    def __init__(self, data, batch_size, learning_rate, units_sentence_encoder, word_emb_size, dropout_probability):
        self.last_epoch = 0
        self.best_epoch = 0
        self.best_pr_area = 0
        self.epochs_since_best = 0
        self.precision_list = []
        self.recall_list = []
        self.f1_measure_list = []
        self.pr_area_list = []
        self.drops = []

        print "Building Model..."

        double_units = 2 * units_sentence_encoder
        target_values = T.ivector('target_output')

        # Input Word embedding
        l_in = lasagne.layers.InputLayer(shape=(batch_size, data.max_len), input_var=T.imatrix())  # [B,t]
        l_emb = lasagne.layers.EmbeddingLayer(l_in, data.word_count + 1, word_emb_size)  # [B,t,h_w]

        # Bi-directional LSTM
        #if sentence_encoder.lower() == 'lstm':
        l_mask = lasagne.layers.InputLayer(shape=(batch_size, data.max_len))
        l_fwd = lasagne.layers.LSTMLayer(l_emb, units_sentence_encoder, mask_input=l_mask, grad_clipping=100,
                                             nonlinearity=lasagne.nonlinearities.tanh, backwards=False,
                                             only_return_final=True)
        l_bwd = lasagne.layers.LSTMLayer(l_emb, units_sentence_encoder, mask_input=l_mask, grad_clipping=100,
                                             nonlinearity=lasagne.nonlinearities.tanh, backwards=True,
                                             only_return_final=True)
        x_i = lasagne.layers.ConcatLayer([l_fwd, l_bwd])  # [B,2h]

        # Dropout
        l_dropout = lasagne.layers.DropoutLayer(x_i, dropout_probability)  # [B,2h]

        # Softmax over O_r
        # [B,|L|]
        self.l_out = lasagne.layers.DenseLayer(l_dropout, num_units=data.rel_count,
                                               nonlinearity=lasagne.nonlinearities.softmax)

        # Final
        network_output = lasagne.layers.get_output(self.l_out)
        cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
        updates = lasagne.updates.adagrad(cost,
                                          lasagne.layers.get_all_params(self.l_out, trainable=True),
                                          learning_rate)
        self.train_function = theano.function([l_in.input_var, l_mask.input_var, target_values],
                                                  cost,
                                                  updates=updates,
                                                  allow_input_downcast=True)
        outputs_argmax = T.argmax(network_output, axis=1)
        self.compute_outputs = theano.function([l_in.input_var, l_mask.input_var],
                                               network_output, allow_input_downcast=True)
        self.compute_outputs_argmax = theano.function([l_in.input_var, l_mask.input_var],
                                               outputs_argmax, allow_input_downcast=True)

    def train(self, data, train_batches):
        print '  Training...'
        for batch in xrange(0, train_batches):
            if batch % 5 == 0:
                print batch,

            mentions = data.train_used[batch][0]
            masks = data.train_used[batch][1]
            relations_sentences = data.train_used[batch][2]

            self.train_function(mentions, masks, relations_sentences)

    def evaluate(self, data, num_batches, validating):
        """
        Computes precision, recall, f1 and p/r curve.
        all_relations refers to a matrix containing a vector of all possible relations for each entry in the large batch
        """
        print '  Evaluating...'

        # Process batches
        for batch in xrange(0, num_batches):
            if validating:
                mentions = data.validation_used[batch][0]
                masks = data.validation_used[batch][1]
                relation_batch = data.validation_used[batch][2]
            else:
                mentions = data.test_used[batch][0]
                masks = data.test_used[batch][1]
                relation_batch = data.test_used[batch][2]
            if batch == 0:
                outputs_argmax = self.compute_outputs_argmax(mentions, masks)
                gold = relation_batch
            else:
                outputs_argmax = np.hstack((outputs_argmax, self.compute_outputs_argmax(mentions, masks)))
                gold = np.hstack((gold, relation_batch))

        if not validating:
            print "GOLD: ", gold
            print "OUTPUTS: ", outputs_argmax

        # Loop through results, compute precision, recall and f1 measure
        none_index = data._rel_dict[NEGATIVE_LABEL]  # Should be 0, though
        precision = []
        recall = []
        f1_measure = []
        true_pos = 0.  # Number of correct predictions
        total_gold_pos = 0.  # Total number of non-negatives
        for i in xrange(0, len(outputs_argmax)):
            if gold[i] != none_index:
                total_gold_pos += 1
        for i in xrange(0, len(outputs_argmax)):
            if gold[i] != none_index:
                if outputs_argmax[i] == gold[i]:
                    true_pos += 1.
            if total_gold_pos == 0:
                p = 0.
                r = 0.
                f = 0.
            else:
                p = float(true_pos) / (float(i) + 1.)
                r = float(true_pos) / float(total_gold_pos)
                if p + r == 0.:
                    f = 0.
                else:
                    f = (2. * p * r / (p + r))
            precision.append(p)
            recall.append(r)
            f1_measure.append(f)

        print "True pos: ", true_pos
        print "Total gold pos: ", total_gold_pos
        print "Total: ", i+1

        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.f1_measure_list.append(f1_measure)

        pr_area = metrics.auc(recall, precision)
        self.pr_area_list.append(pr_area)

        if pr_area >= self.best_pr_area:
            self.best_pr_area = pr_area
            self.best_epoch = self.last_epoch
            self.epochs_since_best = 0
        else:
            self.epochs_since_best += 1

        if not validating:
            print "Precision: ", precision
            print "Recall: ", recall
            print "F1: ", f1_measure
            print "AUC: ", pr_area

    def label_drop(self, data, num_batches, top_true, bottom_false, check):
        print "  Label dropping..."
        num_removed = 0

        # Process batches, get distributions
        for batch in xrange(0, num_batches):
            mentions = data.validation_used[batch][0]
            masks = data.validation_used[batch][1]
            relation_batch = data.validation_used[batch][2]
            if batch == 0:
                mention_stack = mentions
                outputs = self.compute_outputs(mentions, masks)
                golds = relation_batch
            else:
                mention_stack = np.vstack((mention_stack, mentions))
                outputs = np.vstack((outputs, self.compute_outputs(mentions, masks)))
                golds = np.hstack((golds, relation_batch))

        # For each mention and their distribution
        for m in xrange(0, len(mention_stack)):
            ment = mention_stack[m]
            distr = outputs[m]
            gold = golds[m]
            negate = negations_for_label(gold)

            # Get highest ranking labels
            sort = distr.argsort()
            labels_top = sort[-top_true:][::-1]
            labels_bottom = sort[:bottom_false]

            # Check whether the 'gold' label is in the bottom_false
            gold_bottom = False
            for i in xrange(0, bottom_false):
                if labels_bottom[i] == gold:
                    gold_bottom = True

            # Check whether all the top labels negate the 'gold' label
            negate_top = True
            if check:
                for i in xrange(0, top_true):
                    if labels_top[i] not in negate:
                        negate_top = False
                        break

            # If the gold label is in the bottom, and the top only contains negation labels
            if negate_top and not gold_bottom:

                # If it has multiple relations left, remove the relation
                key = "_".join([str(x) for x in [ment]])
                rels_for_ment = data.ment_rels[key]
                if len(rels_for_ment) > 1:
                    try:
                        rels_for_ment.remove(gold)
                    except:
                        print 'not in rels'
                        # Note: gold is sometimes not in rels_for_ment.  Happens rarely.
                        # It appears some sentences appear twice with different labels
                    data.ment_rels[key] = rels_for_ment
                    data.train.pop(key+str(gold))
                    num_removed += 1

        print 'num labels removed: ', num_removed
        self.drops.append(num_removed)

    def save(self, path):
        print('Saving Model...')

        # Weights
        np.savez(path + '_weights.npz', *lasagne.layers.get_all_param_values(self.l_out))

        # Stats
        stats = ModelStats(self.last_epoch, self.best_epoch, self.best_pr_area, self.epochs_since_best,
                           self.precision_list, self.recall_list, self.f1_measure_list, self.pr_area_list, self.drops)
        f = open(path + '_stats.pk', 'wb')
        pickle.dump(stats, f, 2)

    def load(self, path):
        print('Loading existing Model...')

        # Weights
        with np.load(path + '_weights.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.l_out, param_values)

        # Stats
        stats = pickle.load(open(path + '_stats.pk', 'rb'))
        self.last_epoch = stats.last_epoch
        self.best_epoch = stats.best_epoch
        self.best_pr_area = stats.best_pr_area
        self.epochs_since_best = stats.epochs_since_best
        self.precision_list = stats.precision_list
        self.recall_list = stats.recall_list
        self.f1_measure_list = stats.f1_measure_list
        self.pr_area_list = stats.pr_area_list
        self.drops = stats.drops
