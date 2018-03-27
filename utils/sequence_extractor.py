from utils.special_tokens_specification import SpecialTokensSpecification
from utils.special_token_provisioner import SpecialTokenProvisioner
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize.casual import TweetTokenizer
import string
import emoji

class SequenceExtractor:
    def __init__(self, tokenizer_level, use_special_tokens=True, special_tokens_spec=None):
        assert tokenizer_level in ['character', 'word', 'socialmediatokens', 'sentence', 'oracletokens']
        
        self.tokenizer_level = tokenizer_level
        self.special_tokens_spec = self.provision_special_tokens(
            use_special_tokens, special_tokens_spec
        )
        
        self.tokenizer = self.get_tokenizer(self.tokenizer_level)
        self.sep_char = self.get_sep_char(self.tokenizer_level)
        self.join_op = self.get_join_op(self.tokenizer_level)
        
    def provision_special_tokens(self, use_special_tokens, special_tokens_spec):
        if use_special_tokens:
            if special_tokens_spec is None:
                provisioner = SpecialTokenProvisioner(self.tokenizer_level)
                return provisioner.special_tokens_specification
            elif type(special_tokens_spec) is SpecialTokensSpecification:
                return special_tokens_spec
        else:
            return None

    def clean_ascii_line(self, s):
        s = s.replace(u'\ufeff', '')
        ascii_str = ''.join([c for c in s if ord(c) < 128])
        return ascii_str
        
    def character_tokenizer(self, s):
        ascii_chars_str = self.clean_ascii_line(s)
        chars = list(ascii_chars_str)
        return chars

    def sentence_tokenizer(self, s):
        ascii_chars_str = self.clean_ascii_line(s)
        sentences = sent_tokenize(ascii_chars_str)
        return sentences
    
    def word_workenizer(self, s):
        ascii_chars_str = self.clean_ascii_line(s)
        words = word_tokenize(ascii_chars_str)
        return words
    
    def socialmedia_tokenizer(self, s):
        ascii_chars_str = self.clean_ascii_line(s)
        tokens = TweetTokenizer().tokenize(ascii_chars_str)
        return tokens

    def oracle_tokenizer(self, s):
        ascii_chars_str = self.clean_ascii_line(s)
        oracle_tokens = ascii_chars_str.split()
        return oracle_tokens

    def get_tokenizer(self, tokenizer_level):
        if tokenizer_level == 'character':
            return self.character_tokenizer
        elif tokenizer_level == 'word':
            return self.word_workenizer
        elif tokenizer_level == 'socialmediatokens':
            return self.socialmedia_tokenizer
        elif tokenizer_level == 'sentence':
            return self.sentence_tokenizer
        elif tokenizer_level == 'oracletokens':
            return self.oracle_tokenizer
        
    def get_sep_char(self, tokenizer_level):
        if tokenizer_level == 'character':
            return ''
        elif tokenizer_level == 'word':
            return ' '
        elif tokenizer_level == 'socialmediatokens':
            return ' '
        elif tokenizer_level == 'sentence':
            return ' '
        elif tokenizer_level == 'oracletokens':
            return ' '
    
    def char_with_sep(self, token, i, seq_len):
        return self.sep_char + token
        
    def word_with_sep(self, token, i, seq_len):
        nonpunctuation = string.ascii_letters + \
                         string.digits + \
                         '-' + '@' + '#' + '$' + '%' + '&' + '(' + '\'' + '\"'

        whitelisted_traits = [all(x in set(nonpunctuation) for x in token),
                              (len(set(token)) > 1),
                              (token[0] == ":" and len(token) == 2),
                              (token[0] == ":" and len(token) == 3),
                              (token[0] == ";" and len(token) == 2),
                              (token[0] == ";" and len(token) == 3)]

        blacklisted_traits = [(len(token) == 3 and token[0] == "'"),
                              (token == "n't"),
                              (token == "'m"),
                              (token == "'s"),
                              (token[0] == "'" and len(token) > 1 and len(token) <= 3)]

        token_is_wordlike = any(whitelisted_traits) and not any(blacklisted_traits)
        
        if i == 0:
            return token
        elif i > 0 and (token_is_wordlike or self.special_tokens_spec.contains(token)):
            return self.sep_char + token
        else:
            return token

    def socialmediatoken_with_sep(self, token, i, seq_len):
        if i == 0:
            return token
        elif token in emoji.UNICODE_EMOJI:
            return self.sep_char + token
        elif len(token) > 1 and len(token) < 4 and token[0] in [";", ":"]:
            return self.sep_char + token
        else:
            return self.word_with_sep(token, i, seq_len)
    
    def sentence_with_sep(self, token, i, seq_len):
        if i < seq_len-1:
            return token + self.sep_char
        elif i == seq_len-1:
            return token + self.sep_char.strip(" ")

    def oracletoken_with_sep(self, token, i, seq_len):
        if i == 0:
            return token
        elif i < seq_len-1:
            return self.sep_char + token
        else:
            return token
        
    def abstract_join_op(self, tokens, op):
        return ''.join([op(t,i,len(tokens)) for i, t in enumerate(tokens, 0)])
    
    def get_join_op(self, tokenizer_level):
        ops = {
            'character': lambda tokens: self.abstract_join_op(tokens, self.char_with_sep),
            'word':      lambda tokens: self.abstract_join_op(tokens, self.word_with_sep),
            'socialmediatokens': lambda tokens: self.abstract_join_op(tokens, self.socialmediatoken_with_sep),
            'sentence':  lambda tokens: self.abstract_join_op(tokens, self.sentence_with_sep),
            'oracletokens': lambda tokens: self.abstract_join_op(tokens, self.oracletoken_with_sep)
        }
        
        return ops[tokenizer_level]

    def tokenize(self, s):
        return self.tokenizer(s)
    
    def join(self, tokens, drop_special_tokens=False):
        if drop_special_tokens:
            return self.join_op([t for t in tokens if not self.special_tokens_spec.contains(t)])
        else:
            return self.join_op(tokens)