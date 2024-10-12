# -*- coding:utf-8 -*-

import torch
from torch import nn

from config import Config
from data_utils import DataUtils
from generator import Generator


def accuracy(model_out: torch.Tensor, tgt_out: torch.Tensor, PAD_INDEX: int) -> tuple[float, int, int]:
    y_pred = model_out.transpose(0, 1).argmax(axis=2).reshape(-1)
    tgt_out = tgt_out.transpose(0, 1).reshape(-1)
    acc = y_pred.eq(tgt_out)
    mask = torch.logical_not(tgt_out.eq(PAD_INDEX))
    acc = acc.logical_and(mask)
    correct = acc.sum().item()
    total = mask.sum().item()
    return float(correct) / total, correct, total


def evaluate(config_: Config, test_iter, model: Generator, data_utils: DataUtils) -> float:
    model.eval()
    correct, totals = 0, 0
    for _, (src, tgt) in enumerate(test_iter):
        src = src.to(config_.device)
        tgt = tgt.to(config_.device)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = data_utils.gen_mask(src, tgt_input,
                                                                                     device=config_.device)

        model_out = model(src=src,
                          tgt=tgt_input,
                          src_mask=src_mask,
                          tgt_mask=tgt_mask,
                          src_key_padding_mask=src_padding_mask,
                          tgt_key_padding_mask=tgt_padding_mask,
                          memory_key_padding_mask=src_padding_mask)
        tgt_out = tgt[1:, :]
        _, c, t = accuracy(model_out, tgt_out, data_utils.PAD_INDEX)
        correct += c
        totals += t
    model.train()
    return float(correct) / totals


class CustomSchedule(nn.Module):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.step = 1.

    def __call__(self) -> float:
        arg1 = self.step ** -0.5
        arg2 = self.step * (self.warmup_steps ** -1.5)
        self.step += 1.
        return (self.d_model ** -0.5) * min(arg1, arg2)


def train(config_: Config) -> None:
    max_accuracy_on_test: float = 0

    print("Loading data.")
    data_utils = DataUtils()
    train_iter, test_iter = data_utils.data_iter()

    print("Initializing model.")
    generator = Generator(vocab_size=data_utils.vocab_len(),
                          d_model=config_.d_model,
                          nhead=config_.nhead,
                          num_encoder_layers=config_.num_encoder_layers,
                          num_decoder_layers=config_.num_decoder_layers,
                          dim_feedforward=config_.dim_feed_forward,
                          dropout=config_.dropout,
                          device=config_.device)
    for p in generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    loss_fn = nn.CrossEntropyLoss(ignore_index=data_utils.PAD_INDEX)
    learning_rate = CustomSchedule(d_model=config_.d_model)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=0., betas=(config_.beta1, config_.beta2), eps=config_.eps,
                                  weight_decay=0.01)

    print("Start training.")
    for epoch in range(0, config_.num_epochs):
        for index, (src, tgt) in enumerate(train_iter):
            # src: a tensor with shape[max{src_length}, batch_size]
            # tgt: a tensor with shape[max{src_length} + 2, batch_size]
            src = src.to(config_.device)
            tgt_in = tgt[:-1, :]
            tgt_in = tgt_in.to(config_.device)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask \
                = data_utils.gen_mask(src, tgt_in, device=config_.device)
            model_out = generator(src=src,
                                  tgt=tgt_in,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=src_padding_mask)
            optimizer.zero_grad()
            tgt_out = tgt[1:, :]
            tgt_out = tgt_out.to(config_.device)
            loss = loss_fn(
                model_out.reshape(-1, model_out.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            lr = learning_rate()
            for p in optimizer.param_groups:
                p["lr"] = lr
            optimizer.step()

            if index % 10 == 0:
                acc, _, _ = accuracy(model_out, tgt_out, data_utils.PAD_INDEX)
                print(f"Epoch: {epoch}, Batch: {index}/{len(train_iter)}")
                print(f"    Loss: {loss.item()}, Accuracy: {acc:.3f}")

        if epoch % config_.model_save_period == 0:
            acc = evaluate(config_, test_iter, generator, data_utils)
            print("acc on test {:.3f}, max acc on test {:.3f}".format(
                acc, max_accuracy_on_test))
            if acc > max_accuracy_on_test:
                print("save")
                max_accuracy_on_test = acc
                torch.save(generator.state_dict(), config_.model_save_path)


if __name__ == "__main__":
    config = Config()
    train(config_=config)
