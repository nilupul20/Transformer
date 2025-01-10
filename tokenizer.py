from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel, WordPiece
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from transformer_config import TransformerConfig


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def get_tokenizer(cfg:TransformerConfig, dataset, lang) -> Tokenizer:
    tokenizer_path = Path(cfg.tokenizer_file.format(lang))
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