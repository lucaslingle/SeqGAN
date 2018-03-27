from utils.special_tokens_specification import SpecialTokensSpecification

class SpecialTokenProvisioner:
    def __init__(self, tokenizer_level):
        self.tokenizer_level = tokenizer_level
        self.special_tokens_specification = self.get_special_tokens(self.tokenizer_level)
        
    def get_special_tokens(self, tokenizer_level):
        assert tokenizer_level in ['character', 'word', 'socialmediatokens', 'sentence', 'oracletokens']
        specification = None
        
        if tokenizer_level == 'character':
            specification = SpecialTokensSpecification(
                go_token = '\x01',
                unk_token = '\x02',
                pad_token = '\x03',
                eos_token = '\x04'
            )
        elif tokenizer_level == 'word':
            specification = SpecialTokensSpecification(
                go_token = '_GO',
                unk_token = '_UNK',
                pad_token = '_PAD',
                eos_token = '_EOS'
            )
        elif tokenizer_level == 'socialmediatokens':
            specification = SpecialTokensSpecification(
                go_token = '_GO',
                unk_token = '_UNK',
                pad_token = '_PAD',
                eos_token = '_EOS'
            )
            
        return specification