import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics 
from pathlib import Path
from tqdm import tqdm
import argparse

from helpers import get_logger, get_checkpoint_path, get_console_width
from dataset import get_data, casual_mask
from model import build_model, Transformer
from transformer_config import TransformerConfig
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

def greedy_decode(model:Transformer, encoder_input, encoder_mask, tokenizer_tgt:Tokenizer, tgt_seq_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(encoder_input, encoder_mask)

    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(encoder_input).to(device)
    while True:
        if decoder_input.size(1) == tgt_seq_len:
            break

        decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

        prob = model.project(out[:,-1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1,1).type_as(encoder_input).fill_(next_word.item()).to(device)
            ],
            dim=1
        )

        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)


def run_validation(
        model: Transformer, 
        val_dataloader: DataLoader,
        tokenizer_tgt: Tokenizer, 
        tgt_seq_len,
        device,
        print_msg,
        global_step: int,
        writer: SummaryWriter,
        num_examples=4
    ):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []


    console_width = get_console_width()

    with torch.no_grad():
        for batch in val_dataloader:
            count += 1

            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, tgt_seq_len, device)


            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            pred_text = tokenizer_tgt.decode(output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(pred_text)

            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{pred_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break
    
    if writer:
        # Character Error Rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("Validation CER", cer, global_step)
        writer.flush()

        # Word Error Rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("Validation WER", wer, global_step)
        writer.flush()

        # BLEU
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("Validation BLEU", bleu, global_step)
        writer.flush()


def train(cfg:TransformerConfig):
    models_dir =  Path('runs', cfg.model_folder)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(str(models_dir), "train.log")

    if cfg.use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logger.info(f"device: {device}")

    # Dataloaders and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_data(cfg)
    logger.info(f"Train dataset size: {len(train_dataloader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataloader.dataset)}")

    # Transformer Model
    model = build_model(cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    logger.info(f"Initialized the model:\n{model}")

    writer = SummaryWriter(str(models_dir))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-9)

    initial_epoch = 0
    global_step = 0

    # Preload model at a specified epoch
    if cfg.preload:
        checkpoint_name = get_checkpoint_path(cfg, cfg.preload)
        checkpoint = torch.load(checkpoint_name)

        model.load_state_dict(checkpoint['model_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']

        logger.info(f"Preloading model from {checkpoint_name}")
    else:
        logger.info("Model will train from scratch.")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    # Training
    logger.info("Training Starts...")
    for epoch in range(initial_epoch, cfg.num_epochs):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Epoch: {epoch:03d}/{cfg.num_epochs:03d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        run_validation(
            model, 
            val_dataloader,
            tokenizer_tgt, 
            cfg.tgt_seq_len, 
            device, 
            lambda msg: batch_iterator.write(msg), 
            global_step, 
            writer
        )

        # save checkpoint at each epoch
        model_filename = get_checkpoint_path(cfg, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        logger.info(f"Model Checkpoint saved at epoch {epoch}: {model_filename}")

    logger.info("Training Completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = TransformerConfig.from_yaml(args.config)

    train(cfg)