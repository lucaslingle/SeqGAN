import pandas as pd
import csv
from utils.special_tokens_specification import SpecialTokensSpecification
from utils.special_token_provisioner import SpecialTokenProvisioner
from utils.sequence_extractor import SequenceExtractor
from utils.vocab_dictionary import VocabDictionary

class DatasetReader:
    def __init__(self, fp, file_sepchar, file_columns, text_colname, sequence_extractor,
                 min_seq_length=5, max_seq_length=100, vocab_dict=None):

        self.fp = fp
        self.file_sepchar = file_sepchar
        self.file_columns = file_columns

        self.text_colname = text_colname
        self.seq_extractor = sequence_extractor

        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length

        self.vocab_dict = vocab_dict

    def load(self):
        self.reprocess()

        if self.vocab_dict is None:
            lines = self.standardized_df[self.text_colname].tolist()
            self.vocab_dict = VocabDictionary(
                seq_extractor=self.seq_extractor,
                max_seq_length=self.max_seq_length,
                drop_freq_thresh=(0 if self.seq_extractor.tokenizer_level == 'oracletokens' else 2)
            )
            self.vocab_dict.init_natural(lines=lines)

    def reprocess(self):

        df = pd.read_csv(self.fp, sep=self.file_sepchar, names=self.file_columns,
                         quoting=csv.QUOTE_NONE, error_bad_lines=False, encoding='utf-8')

        self.filtered_df = df.copy()
        self.filtered_df = self.filtered_df[
            df[self.text_colname].apply(lambda line: self.admissable(line))
        ]

        self.standardized_df = self.filtered_df.copy()
        self.standardized_df[self.text_colname] = self.filtered_df[self.text_colname].apply(
            lambda line: self.standardize(line)
        )

    def admissable(self, line):
        tokens = self.seq_extractor.tokenize(line)
        if self.min_seq_length <= len(tokens) and len(tokens) <= self.max_seq_length:
            return True
        else:
            return False
        
    def standardize(self, line):
        tokens = self.seq_extractor.tokenize(line)
        pad_token = self.seq_extractor.special_tokens_spec.pad_token
        
        if self.max_seq_length - len(tokens) > 0:
            tokens += [pad_token for _ in range(self.max_seq_length-len(tokens))]

        joined = self.seq_extractor.join(tokens)
        return joined