from utils.special_tokens_specification import SpecialTokensSpecification
from utils.special_token_provisioner import SpecialTokenProvisioner
from utils.sequence_extractor import SequenceExtractor
from collections import Counter

class VocabDictionary():
    def __init__(self, seq_extractor, max_seq_length=20, drop_freq_thresh=10):
        self.seq_extractor = seq_extractor
        self.special_tokens_spec = seq_extractor.special_tokens_spec
        self.max_seq_length = max_seq_length
        self.drop_freq_thresh = drop_freq_thresh

    def init_natural(self, lines):

        self.counter = Counter()

        for line in lines:
            line_tokens = self.seq_extractor.tokenize(line)
            self.counter.update(line_tokens)

        self.counter = Counter({k: v for k, v in self.counter.items() if v >= self.drop_freq_thresh})

        self.ordered_vocab_list = []
        self.ordered_vocab_list.append(self.special_tokens_spec.go_token)  # _GO token must be index 0
        self.ordered_vocab_list.append(self.special_tokens_spec.unk_token)  # _UNK token must be index 1
        self.ordered_vocab_list.append(self.special_tokens_spec.pad_token)  # _PAD token must be index 2
        self.ordered_vocab_list.append(self.special_tokens_spec.eos_token)  # _EOS token must be index 3

        nonspecial_chars = [key for key in list(self.counter.keys()) if not self.special_tokens_spec.contains(key)]
        self.ordered_vocab_list.extend(sorted(nonspecial_chars))

        self.vocab_dict = {w: i for i, w in enumerate(self.ordered_vocab_list, 0)}
        self.int_to_token_dict = {i: w for i, w in enumerate(self.ordered_vocab_list, 0)}

    def init_oracle(self, oracle_vocab_size):
        self.ordered_vocab_list = [str(i) for i in range(0, oracle_vocab_size)]

        self.vocab_dict = {w: i for i, w in enumerate(self.ordered_vocab_list, 0)}
        self.int_to_token_dict = {i: w for i, w in enumerate(self.ordered_vocab_list, 0)}


    def lookup(self, token):
        if token in self.vocab_dict:
            return self.vocab_dict[token]
        else:
            return self.vocab_dict[self.special_tokens_spec.unk_token]

    def reverse_lookup(self, word_id):
        return self.int_to_token_dict[word_id]

    def get_length(self):
        return len(self.vocab_dict)