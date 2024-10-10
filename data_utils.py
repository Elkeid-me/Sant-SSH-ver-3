# -*- coding:utf-8 -*-

import json
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from config import tokenizer, Config
from singleton import Singleton
from vocab import Vocab

config = Config()


class DataUtils(Singleton):
    def __init__(self):
        self._special = ["/", "%", "^", "#"]
        # / for unk, % for pad, ^ for bos, # for eos
        counter = Counter()
        for path in [*config.train_file_path, *config.test_file_path]:
            with open(path, mode='r', encoding="utf8") as fin:
                tmp_list = json.load(fin)
            for sentence in tmp_list:
                counter.update(tokenizer(sentence))

        print("    Building vocab")
        self._vocab: Vocab = Vocab(config)
        self.PAD_INDEX = self.char_to_int("%")
        self.BOS_INDEX = self.char_to_int("^")
        self.EOS_INDEX = self.char_to_int("#")

    def char_to_int(self, char: str) -> int:
        return self._vocab.char_to_int(char)

    def int_to_char(self, index: int) -> str:
        return self._vocab.int_to_char(index)

    def _data_process(self, load_path: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        ret_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        with open(load_path[0], mode='r', encoding="utf8") as in_fin:
            in_data: list[str] = json.load(in_fin)
        with open(load_path[1], mode='r', encoding="utf8") as in_fin:
            out_data: list[str] = json.load(in_fin)

        length: int = len(in_data)
        for index, (in_sentence, out_sentence) in enumerate(zip(in_data, out_data)):
            in_tensor = torch.tensor([self.char_to_int(char)
                                     for char in tokenizer(in_sentence)])
            out_tensor = torch.tensor(
                [self.BOS_INDEX, *[self.char_to_int(char) for char in tokenizer(out_sentence)], self.EOS_INDEX])
            ret_data.append((in_tensor, out_tensor))

            if index % 10000 == 0:
                print(f"            process {index}/{length}")

        return ret_data

    def _generate_batch(self, data_batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        in_batch = []
        out_batch = []
        for in_sentence, out_sentence in data_batch:
            in_batch.append(in_sentence)
            out_batch.append(out_sentence)
        in_batch = pad_sequence(in_batch, padding_value=self.PAD_INDEX)
        out_batch = pad_sequence(out_batch, padding_value=self.PAD_INDEX)
        return in_batch, out_batch

    def data_iter(self) -> tuple[DataLoader, DataLoader]:
        print("    Generating iter.")
        print("        Processing train data.")
        train_data = self._data_process(load_path=config.train_file_path)
        print("        Processing test data.")
        test_data = self._data_process(load_path=config.test_file_path)
        train_iter = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True,
                                collate_fn=self._generate_batch)
        test_iter = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True,
                               collate_fn=self._generate_batch)
        return train_iter, test_iter

    def gen_mask(self, src: torch.Tensor, tgt: torch.Tensor, device) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_len = src.shape[0]
        tgt_len = tgt.shape[0]
        src_mask: torch.Tensor = torch.zeros(
            (src_len, src_len)).type(torch.bool)
        tgt_mask: torch.Tensor = torch.nn.Transformer.generate_square_subsequent_mask(
            tgt_len)
        src_padding_mask: torch.Tensor = (
            src == self.PAD_INDEX).transpose(0, 1)
        tgt_padding_mask: torch.Tensor = (
            tgt == self.PAD_INDEX).transpose(0, 1)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        src_padding_mask = src_padding_mask.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def vocab_len(self) -> int:
        return len(self._vocab)
