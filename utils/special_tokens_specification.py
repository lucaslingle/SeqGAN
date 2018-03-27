class SpecialTokensSpecification:
    def __init__(self, go_token, unk_token, pad_token, eos_token):
        self.go_token = go_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.special_tokens = [self.go_token, self.unk_token, self.pad_token, self.eos_token]
        
    def contains(self, token):
        return (token in self.special_tokens)