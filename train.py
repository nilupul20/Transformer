from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, casual_mask

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, WordPiece
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def get_tokenizer(cfg, dataset, lang) -> Tokenizer:
    tokenizer_path = Path(cfg['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency=2, 
            show_progress=True
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

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
        cfg['seq_len']
    )
    val_dataset = BilingualDataset(
        val_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        cfg['lang_src'],
        cfg['lang_tgt'],
        cfg['seq_len']
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
    val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt