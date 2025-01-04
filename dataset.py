import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src:Tokenizer, tokenizer_tgt:Tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_src.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # both [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # only [SOS]

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        return super().__getitem__(index)