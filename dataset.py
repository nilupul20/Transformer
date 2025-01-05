import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from datasets import load_dataset

from tokenizer import get_tokenizer

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src:Tokenizer, tokenizer_tgt:Tokenizer, src_lang, tgt_lang, src_seq_len, tgt_seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data_point = self.dataset[index]
        src_text = data_point['translation'][self.src_lang]
        tgt_text = data_point['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.src_seq_len - len(enc_input_tokens) - 2 # both [SOS] and [EOS]
        dec_num_padding_tokens = self.tgt_seq_len - len(dec_input_tokens) - 1 # only [SOS]

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # make sure add upto seq_len
        assert encoder_input.size(0) == self.src_seq_len
        assert decoder_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len

        return {
            "encoder_input": encoder_input, # seq_len
            "decoder_input": decoder_input, # seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1, seq_len) $ (1, seq_len, seq_len)
            "label": label, # seq_len
            "src_text": src_text,
            "tgt_text": tgt_text

        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_dataset(cfg:dict):
    dataset_raw = load_dataset('opus_books', f'{cfg["lang_src"]}-{cfg["lang_tgt"]}', split='train')

    tokenizer_src = get_tokenizer(cfg, dataset_raw, cfg['lang_src'])
    tokenizer_tgt = get_tokenizer(cfg, dataset_raw, cfg['lang_tgt'])

    val_split = cfg.get('val_split', 0.9)
    train_size = int(val_split * len(dataset_raw))
    val_size = len(dataset_raw) - train_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_size, val_size] )

    train_dataset = BilingualDataset(
        train_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        cfg['lang_src'],
        cfg['lang_tgt'],
        cfg['src_seq_len'],
        cfg['tgt_seq_len']
    )
    val_dataset = BilingualDataset(
        val_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        cfg['lang_src'],
        cfg['lang_tgt'],
        cfg['src_seq_len'],
        cfg['tgt_seq_len']
    )

    max_len_src = 0
    max_len_tgt = 0
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][cfg['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][cfg['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"Maximum source sentence length: {max_len_src}")
    print(f"Maximum target sentence length: {max_len_tgt}")

    batch_size=cfg.get('batch_size', 32)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt