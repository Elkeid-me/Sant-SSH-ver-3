# -*- coding:utf-8 -*-

import json
from config import Config
from singleton import Singleton


class Vocab(Singleton):
    def __init__(self, config_: Config):
        with open(config_.vocab_path[0], mode="r", encoding="utf8") as fin:
            self._char_to_int: dict[str, int] = json.load(fin)
        with open(config_.vocab_path[1], mode="r", encoding="utf8") as fin:
            self._int_to_char: list[str] = json.load(fin)
        self._len: int = len(self._int_to_char)
        self.marks: list[str] = ['，', '：', '！', '；', '？']
        self.PAD_INDEX = self.char_to_int("%")
        self.BOS_INDEX = self.char_to_int("^")
        self.EOS_INDEX = self.char_to_int("#")

    def char_to_int(self, char: str) -> int:
        if char in self._char_to_int:
            return self._char_to_int[char]
        return 0

    def int_to_char(self, index: int) -> str:
        if 0 <= index < self._len:
            return self._int_to_char[index]
        return '/'

    def __len__(self) -> int:
        return self._len
