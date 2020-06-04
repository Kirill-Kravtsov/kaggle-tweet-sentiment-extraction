import os
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from types import SimpleNamespace


class BERTweetTokenizer():
    
    def __init__(self, pretrained_path="../pretrained/bertweet/"):
        
        self.bpe = fastBPE(SimpleNamespace(bpe_codes=os.path.join(pretrained_path, "bpe.codes")))
        self.vocab = Dictionary()
        self.vocab.add_from_file(os.path.join(pretrained_path, "dict.txt"))
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.pad_token = '<pad>'
        self.cls_token = '<s>'
        self.sep_token = '</s>'
        
    def bpe_encode(self,text):
        return self.bpe.encode(text)
    
    def encode(self,text,add_special_tokens=False):
        subwords = self.bpe.encode(text)
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        return input_ids
    
    def tokenize(self,text):
        return self.bpe_encode(text).split()
    
    def convert_tokens_to_ids(self,tokens):
        input_ids = self.vocab.encode_line(' '.join(tokens), append_eos=False, add_if_not_exist=False).long().tolist()
        return input_ids
    
    #from: https://www.kaggle.com/nandhuelan/bertweet-first-look
    def decode_id(self,id):
        return self.vocab.string(id, bpe_symbol = '@@')
    
    def decode_id_nospace(self,id):
        return self.vocab.string(id, bpe_symbol = '@@ ')
