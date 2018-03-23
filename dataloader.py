import numpy as np
from nltk import word_tokenize
from collections import Counter

class VocabDictionary():
    def __init__(self, data_fp, max_seq_length=20, character_level_model_bool=False, drop_freq_thresh=10):

        self.character_level_model_bool = character_level_model_bool
        self.counter = Counter()
        self.max_seq_length = max_seq_length
        self.drop_freq_thresh = drop_freq_thresh

        self.go_token = '\x01'
        self.unk_token = '\x02'
        self.pad_token = '\x03'
        self.eos_token = '\x04'

        self.special_tokens = [self.go_token, self.unk_token, self.pad_token, self.eos_token]

        def _char_tokenize(line):
            line_tokens_list = [c for c in line if ord(c) < 128]
            return line_tokens_list

        def _word_tokenize(line):
            line_tokens_list = word_tokenize(line)
            return line_tokens_list

        def tokenizerFactory():
            return _char_tokenize if self.character_level_model_bool else _word_tokenize

        self.tokenizer = tokenizerFactory()

        with open(data_fp, 'r') as f:
            for line in f:
                line = line.strip().split("\t")[0]
                line_tokens_list = self.tokenizer(line)
                self.counter.update(line_tokens_list)

        special_tokens_in_original_data = set(self.special_tokens).intersection(list(self.counter.keys()))
        assert len(special_tokens_in_original_data) == 0

        self.counter = Counter({k: v for k, v in self.counter.items() if v > self.drop_freq_thresh})

        self.ordered_vocab_list = []
        self.ordered_vocab_list.append(self.go_token)  # _GO token must be index 0
        self.ordered_vocab_list.append(self.unk_token)  # _UNK token must be index 1
        self.ordered_vocab_list.append(self.pad_token)  # _PAD token must be index 2
        self.ordered_vocab_list.append(self.eos_token)  # _EOS token must be index 3
        self.ordered_vocab_list.extend([
            x for x in self.special_tokens if x not in [self.go_token, self.unk_token, self.pad_token, self.eos_token]
          ]
        )
        self.ordered_vocab_list.extend(sorted(list(self.counter.keys())))

        self.vocab_dict = {w: i for i, w in enumerate(self.ordered_vocab_list, 0)}
        self.int_to_token_dict = {i: w for i, w in enumerate(self.ordered_vocab_list, 0)}

    def lookup(self, token):
        return self.vocab_dict[token] if token in self.vocab_dict else self.vocab_dict[self.unk_token]

    def reverse_lookup(self, word_id):
        return self.int_to_token_dict[word_id]

    def get_length(self):
        return len(self.vocab_dict)


class Gen_Dataloader():
    def __init__(self, batch_size, vocab_dictionary=None,
                 max_seq_length=20, min_seq_length=5, character_level_model_bool=False):
        self.batch_size = batch_size
        self.token_stream = []
        self.vocab_dictionary = vocab_dictionary
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.character_level_model_bool = character_level_model_bool

        def _char_tokenize(line):
            line_tokens_list = [c for c in line if ord(c) < 128]
            return line_tokens_list

        def _word_tokenize(line):
            line_tokens_list = word_tokenize(line)
            return line_tokens_list

        def tokenizerFactory():
            return _char_tokenize if self.character_level_model_bool else _word_tokenize

        self.tokenizer = tokenizerFactory()

    def create_batches(self, data_file):
        self.token_stream = []

        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip().split("\t")[0]
                line = line.split() if self.vocab_dictionary is None else self.tokenizer(line)
                parse_line = [int(x) if self.vocab_dictionary is None else self.vocab_dictionary.lookup(x)
                              for x in line]

                if len(parse_line) == self.max_seq_length:
                    self.token_stream.append(parse_line)
                elif (self.vocab_dictionary is not None) and len(parse_line) < self.max_seq_length and len(parse_line) >= self.min_seq_length:
                    pad_token = self.vocab_dictionary.pad_token
                    pad_int = self.vocab_dictionary.lookup(pad_token)
                    pad_len = self.max_seq_length - len(parse_line)
                    parse_line.extend([pad_int for _ in range(pad_len)])
                    self.token_stream.append(parse_line)

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
    def __init__(self, batch_size, vocab_dictionary=None,
                 max_seq_length=20, min_seq_length=5, character_level_model_bool=False):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.vocab_dictionary = vocab_dictionary
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.character_level_model_bool = character_level_model_bool

        def _char_tokenize(line):
            line_tokens_list = [c for c in line if ord(c) < 128]
            return line_tokens_list

        def _word_tokenize(line):
            line_tokens_list = word_tokenize(line)
            return line_tokens_list

        def tokenizerFactory():
            return _char_tokenize if self.character_level_model_bool else _word_tokenize

        self.tokenizer = tokenizerFactory()

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []

        with open(positive_file, 'r') as fin:
            for line in fin:
                line = line.strip().split("\t")[0]
                line = line.split() if self.vocab_dictionary is None else self.tokenizer(line)
                parse_line = [int(x) if self.vocab_dictionary is None else self.vocab_dictionary.lookup(x)
                              for x in line]

                if len(parse_line) == self.max_seq_length:
                    positive_examples.append(parse_line)
                elif (self.vocab_dictionary is not None) and len(parse_line) < self.max_seq_length and len(parse_line) >= self.min_seq_length:
                    pad_token = self.vocab_dictionary.pad_token
                    pad_int = self.vocab_dictionary.lookup(pad_token)
                    pad_len = self.max_seq_length - len(parse_line)
                    parse_line.extend([pad_int for _ in range(pad_len)])
                    positive_examples.append(parse_line)

        with open(negative_file, 'r') as fin:
            for line in fin:
                line = line.strip().split("\t")[0]
                line = line.split() if self.vocab_dictionary is None else self.tokenizer(line)
                parse_line = [int(x) if self.vocab_dictionary is None else self.vocab_dictionary.lookup(x)
                              for x in line]

                if len(parse_line) == self.max_seq_length:
                    negative_examples.append(parse_line)
                elif (self.vocab_dictionary is not None) and len(parse_line) < self.max_seq_length:
                    pad_token = self.vocab_dictionary.pad_token
                    pad_int = self.vocab_dictionary.lookup(pad_token)
                    pad_len = self.max_seq_length - len(parse_line)
                    parse_line.extend([pad_int for _ in range(pad_len)])
                    negative_examples.append(parse_line)

        # we don't want to permit class imbalances per batch.

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
