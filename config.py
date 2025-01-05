from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epoches": 50,
        "lr": 0.0001,
        "src_seq_len": 350,
        "tgt_seq_len": 350,
        "d_model": 512,
        "h": 8,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "saved_models",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)