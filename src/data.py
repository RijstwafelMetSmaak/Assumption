import google.protobuf.internal.decoder as decoder
import Document_pb2
import numpy as np
import random
from definitions import ROOT_PATH, NEGATIVE_LABEL

class Data:
    """
    Contains a .train, .test  and optionally a .validation set, sorted by entity pairs and relations, where:
    dataset[index][0] = mentions (shape=[max_len])
    dataset[index][1] = mentions masks (shape=[max_len])
    dataset[index][2] = relation (shape=1)
    dataset[index][3] = entity pair (shape=1, concatenated string, ditched when split)

    Validation and batch sizes are not saved. Call split_validation and split_batches separately.

    """

    def __init__(self, max_len):
        print 'Preprocessing new dataset...'
        self._doc_parse = Document_pb2.Relation()  # For parsing
        self._word_dict = {}  # Dictionary for mapping all words to indices
        self._rel_dict = {}  # Dictionary for mapping all relations to indices
        self._rel_dict[NEGATIVE_LABEL] = 0  # Preset the negative label as index 0

        self.word_count = 0  # Index for the latest new word. Can be interpreted as the current vocabulary size
        self.rel_count = 1  # Index for the latest new relation. Can be interpreted as the current number of relations

        self.max_len = max_len  # Maximum sentence length

        self.train = self._preprocess(ROOT_PATH + '/data/train.pb', True)  # Riedel train data
        print '  Training set built.'
        self.test = self._preprocess(ROOT_PATH + '/data/test.pb', False)  # Riedel test data
        print '  Testing set built.'

        self.validation_portion = 0
        self.validation = None

        self.inv_rel_dict = [-1] * len(self._rel_dict.keys())
        for k in self._rel_dict.keys():
            print k
            self.inv_rel_dict[self._rel_dict[k]] = k

    def get_train_values(self):
        self.train_used = self.train.values()

    def shuffle(self):
        print "  Shuffling..."
        np.random.shuffle(self.train_used)
        # np.random.shuffle(self.test)
        if self.validation:
            np.random.shuffle(self.validation)

    def split_validation(self, validation_portion):
        """
        Splits off part of the training data to be used as validation data.
        validation_portion should be a float between 0.0 and 1.0
        """
        validation_size = int(round(validation_portion * (len(self.train_used))))
        self.validation_used = self.train_used[:validation_size]
        self.train_used = self.train_used[validation_size:]

    def split_batches(self, batch_size):
        self.train_used = self.split_list(self.train_used, batch_size)
        self.test_used = self.split_list(self.test, batch_size)
        self.validation_used = self.split_list(self.validation_used, batch_size)

    def split_list(self, list, batch_size):
        new_list = []

        # For each batch
        for i in xrange(0, int(round(len(list) / batch_size))):

            # Combine pairs from the old list
            for j, pair in enumerate((list[i * batch_size:(i + 1) * batch_size])):
                if j == 0:
                    mentions = pair[0]
                    masks = pair[1]
                    relations_sentences = pair[2]
                    # relation_pair = pair[3]
                else:
                    # Note: could do this faster rather than one by one
                    mentions = np.vstack((mentions, np.asarray(pair[0], dtype=np.int32)))
                    masks = np.vstack((masks, np.asarray(pair[1], dtype=np.float32)))
                    relations_sentences = np.hstack((relations_sentences, np.asarray(pair[2], dtype=np.int32)))
                    # relation_pair = np.hstack((relation_pair, np.asarray(pair[3], dtype=np.int32)))

            new_list.append((mentions, masks, relations_sentences))

        return new_list

    def _pad_and_mask(self, ment):
        """Pads the mention with a padding character until size max_len, and creates a corresponding mask"""
        mask = np.zeros(self.max_len, dtype=np.int)
        mask[:len(ment)] = 1
        sentence = np.lib.pad(ment, (0, self.max_len - len(ment)), 'constant', constant_values=0)
        return sentence, mask

    def _preprocess_mentions(self):
        # Note: could do this faster by starting with a zero array and just filling it up. No need for padding
        mentions = None
        masks = None

        # Parse
        for m, ment in enumerate(self._doc_parse.mention):
            ment = ment.sentence.lower().split()
            length = len(ment)
            if length > self.max_len:
                continue

            # Index and pad mentions, create corresponding masks, find distances from entities
            for i, word in enumerate(ment):
                if word not in self._word_dict:
                    self.word_count += 1
                    self._word_dict[word] = self.word_count
                ment[i] = self._word_dict[word]
            ment, mask = self._pad_and_mask(ment)
            ment = np.asarray(ment, dtype=np.int32)
            mask = np.asarray(mask, dtype=np.float32)

            # Add to array
            if mentions is None:
                mentions = ment
                masks = mask
            else:
                mentions = np.vstack((mentions, ment))
                masks = np.vstack((masks, mask))

        return mentions, masks

    def _preprocess_relations(self):
        relations = list(set(self._doc_parse.relType.split(',')))
        for i, rel in enumerate(relations):
            if rel not in self._rel_dict:
                self.rel_count += 1
                self._rel_dict[rel] = self.rel_count - 1
            relations[i] = self._rel_dict[rel]
        return relations

    def _preprocess(self, path, is_train):
        if is_train:
            dataset = {}  # Final dataset to be returned
        else:
            dataset = []
        buff = open(path, 'rb').read()  # For parsing
        size, pos = 0, 0  # For parsing
        rel_pass = 0
        if is_train:
            self.ment_rels = {}

        while True:
            try:
                (size, pos) = decoder._DecodeVarint(buff, pos)
                self._doc_parse.ParseFromString(buff[pos:pos + size])

                pair = self._doc_parse.sourceGuid + "_" + self._doc_parse.destGuid
                mentions, masks = self._preprocess_mentions()

                if mentions is not None:
                    relations = self._preprocess_relations()

                    # Drop a portion of null labels
                    if relations[0] == 0:
                        if rel_pass < 30:
                            rel_pass += 1
                            pos += size
                            continue
                        else:
                            rel_pass = 0

                    if mentions.ndim == 1:
                        mentions = [mentions]
                        masks = [masks]

                    for m, ment in enumerate(mentions):

                        if len(ment) != len(masks[m]):
                            print ment
                            print masks[m]

                        if is_train:
                            key = "_".join([str(x) for x in [ment]])
                            self.ment_rels[key] = relations
                            for rel in relations:
                                dataset[key+str(rel)] = (ment, masks[m], rel, pair)
                        else:
                            for rel in relations:
                                dataset.append((ment, masks[m], rel, pair))

                pos += size
            except IndexError:
                break

        return dataset
