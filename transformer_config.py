import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TransformerConfig:
    src_seq_len: int
    tgt_seq_len: int

    lang_src: str
    lang_tgt: str
    model_folder: str
    model_basename: str
    tokenizer_file: str

    val_split: float = 0.9
    batch_size: int = 8
    num_epochs: int = 20
    lr: float = 0.0001
    use_cuda: bool = False
    preload: str|None = None

    d_model: int = 512
    num_layers = 6
    d_ff: int = 2048
    h: int = 8
    dropout: float = 0.1

    @staticmethod
    def from_yaml(file_path: str):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return TransformerConfig(**data)
    
    def save_to_yaml(self, directory: str):
        file_path = str(Path(directory, 'config.yaml'))
        with open(file_path, 'w') as file:
            yaml.dump(self.__dict__, file, default_flow_style=False)