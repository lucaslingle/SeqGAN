import numpy as np
from nltk import word_tokenize
from collections import Counter
from utils.sequence_extractor import SequenceExtractor

class Gen_Dataloader():
    def __init__(self, vocab_dict, dataset_reader, batch_size):

        self.vocab_dict = vocab_dict
        self.dataset_reader = dataset_reader
        self.batch_size = batch_size

        self.token_stream = []

    def create_batches(self):

        self.dataset_reader.reprocess()

        textcol_series = self.dataset_reader.standardized_df[self.dataset_reader.text_colname]
        textcol_series = textcol_series.apply(
            lambda line: [self.vocab_dict.lookup(x) for x in self.dataset_reader.seq_extractor.tokenize(line)]
        )

        self.token_stream = textcol_series.tolist()
        self.token_stream = [token_ints_list for token_ints_list in self.token_stream
                             if len(token_ints_list) == self.dataset_reader.max_seq_length]

        shuffle_indices = np.random.permutation(np.arange(len(self.token_stream)))
        self.token_stream = np.array(self.token_stream)[shuffle_indices]

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(self.token_stream, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_Dataloader():
    def __init__(self, vocab_dict, positive_dataset_reader, negative_dataset_reader, batch_size):

        self.vocab_dict = vocab_dict
        self.positive_dataset_reader = positive_dataset_reader
        self.negative_dataset_reader = negative_dataset_reader
        self.batch_size = batch_size

        self.token_stream = []

    def load_train_data(self):

        self.positive_dataset_reader.reprocess()
        self.negative_dataset_reader.reprocess()

        # normalize as ints
        positive_examples_series = self.positive_dataset_reader.standardized_df[self.positive_dataset_reader.text_colname]
        positive_examples_series = positive_examples_series.apply(
            lambda line: [self.vocab_dict.lookup(x) for x in self.positive_dataset_reader.seq_extractor.tokenize(line)]
        )
        positive_examples = positive_examples_series.tolist()

        negative_examples_series = self.negative_dataset_reader.standardized_df[self.negative_dataset_reader.text_colname]
        negative_examples_series = negative_examples_series.apply(
            lambda line: [self.vocab_dict.lookup(x) for x in self.negative_dataset_reader.seq_extractor.tokenize(line)]
        )
        negative_examples = negative_examples_series.tolist()

        # make sure no parsing inconsistencies -- gotta have correct length
        positive_examples = [token_ints_list for token_ints_list in positive_examples
                             if len(token_ints_list) == self.positive_dataset_reader.max_seq_length]

        negative_examples = [token_ints_list for token_ints_list in negative_examples
                             if len(token_ints_list) == self.negative_dataset_reader.max_seq_length]

        # Also, we don't want to permit class imbalances per batch.
        L = min(len(positive_examples), len(negative_examples))
        assert L > 0

        positive_examples = positive_examples[0:L]
        negative_examples = negative_examples[0:L]

        positive_sentences = np.array(positive_examples)
        negative_sentences = np.array(negative_examples)

        # Generate labels
        positive_labels = np.array([[0, 1] for _ in positive_examples])
        negative_labels = np.array([[1, 0] for _ in negative_examples])

        # Shuffle the data
        shuffle_indices_positive = np.random.permutation(np.arange(len(positive_labels)))
        shuffle_indices_negative = np.random.permutation(np.arange(len(negative_labels)))

        positive_sentences = positive_sentences[shuffle_indices_positive]
        positive_labels = positive_labels[shuffle_indices_positive]  # we don't actually need this.

        negative_sentences = negative_sentences[shuffle_indices_negative]
        negative_labels = negative_labels[shuffle_indices_negative]  # we don't actually need this.

        # Make batches
        #
        # Each batch will be half positive half negative. So if batch size is 32, we break up the positive data
        # into batches of size 16, and likewise for the negative data.
        # Then we concatenate the two size 16 batches together to get a class-balanced batch.
        #
        num_batch_per_class = int(
            len(positive_labels) // (self.batch_size // 2))  # dataset sizes for pos/neg asserted to be equal
        self.num_batch = num_batch_per_class  # dataset sizes for pos/neg asserted to be equal

        positive_sentences = positive_sentences[:num_batch_per_class * (self.batch_size // 2)]
        negative_sentences = negative_sentences[:num_batch_per_class * (self.batch_size // 2)]

        positive_labels = positive_labels[:num_batch_per_class * (self.batch_size // 2)]
        negative_labels = negative_labels[:num_batch_per_class * (self.batch_size // 2)]

        positive_sentences_batches = np.split(positive_sentences, num_batch_per_class, 0)
        positive_labels_batches = np.split(positive_labels, num_batch_per_class, 0)

        negative_sentences_batches = np.split(negative_sentences, num_batch_per_class, 0)
        negative_labels_batches = np.split(negative_labels, num_batch_per_class, 0)

        self.sentences_batches = [np.concatenate([pos_sent_batch, neg_sent_batch], axis=0) for
                                  (pos_sent_batch, neg_sent_batch)
                                  in list(zip(positive_sentences_batches, negative_sentences_batches))]

        self.labels_batches = [np.concatenate([pos_labels_batch, neg_labels_batch], axis=0) for
                               (pos_labels_batch, neg_labels_batch)
                               in list(zip(positive_labels_batches, negative_labels_batches))]

        self.sentences = np.concatenate(self.sentences_batches, axis=0)
        self.labels = np.concatenate(self.labels_batches, axis=0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
