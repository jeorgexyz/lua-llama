from typing import Optional
import torch
import time
from pathlib import Path
import json 
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Tranformer

class LLaMA:

    def __init__(self, model_args: ModelArgs, device: torch.device, tokenizer: SentencePieceProcessor, model: Tranformer):
        self.model_args = model_args
        self.device = device
        self.tokenizer = tokenizer
        self.model = model_args


        @staticmethod
        def build(checkpoints_dir: str, tokenizer_path)
            