# -*- coding:utf-8 -*-

import torch

from config import Config
from vocab import Vocab
from generator import Generator

if __name__ == "__main__":
    config = Config()
    print("Loading data")
    vocab = Vocab(config_=config)
    print("    Loading model")
    model = Generator(vocab_size=len(vocab),
                      d_model=config.d_model,
                      nhead=config.nhead,
                      num_encoder_layers=config.num_encoder_layers,
                      num_decoder_layers=config.num_decoder_layers,
                      dim_feedforward=config.dim_feed_forward,
                      dropout=config.dropout,
                      device=config.device)
    loaded_paras = torch.load(config.model_save_path, config.device)
    model.load_state_dict(loaded_paras)
    model.eval()
    src = ""
    while True:
        src = input("请输入上联：")
        if src == "114514":
            break
        print(f"上联：{src}")
        print(f"下联：{model.greedy_decode(vocab_=vocab, config_=config, src_=src)}")
