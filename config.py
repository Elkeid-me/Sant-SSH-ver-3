# -*- coding:utf-8 -*-

from torch.cuda import is_available

from singleton import Singleton


def tokenizer(s: str) -> list[str]:
    return s.split()


class Config(Singleton):
    def __init__(self):
        self.on_cluster: bool = False
        self.debug: bool = True
        self.use_cuda: bool = True

        # dataset path and save path
        if self.on_cluster:
            self.train_file_path: list[str] = [
                "/dataset/train_in.json", "/dataset/train_out.json"]
            self.test_file_path: list[str] = [
                "/dataset/test_in.json", "/dataset/test_out.json"]
            if self.debug:
                self.model_save_path: str = "/code/model/model.pkl"
            else:
                self.model_save_path: str = "/model/model.pkl"
        else:
            self.train_file_path: list[str] = [
                "dataset/train_in.json", "dataset/train_out.json"]
            self.test_file_path: list[str] = [
                "dataset/test_in.json", "dataset/test_out.json"]
            self.model_save_path: str = "model/model.pkl"

        self.vocab_path: list[str] = [
            "dataset/vocab_char_to_int.json", "dataset/vocab_int_to_char.json"]

        self.d_model: int = 256
        self.nhead: int = 8
        self.num_encoder_layers: int = 4
        self.num_decoder_layers: int = 4
        self.dim_feed_forward: int = 512
        self.dropout: float = 0.1
        self.device: str = "cuda:0" if is_available() and self.use_cuda else "cpu"
        self.batch_size: int = 512
        self.beta1: float = 0.9
        self.beta2: float = 0.98
        self.eps: float = 1e-9
        self.num_epochs: int = 200
        self.model_save_period: int = 2
