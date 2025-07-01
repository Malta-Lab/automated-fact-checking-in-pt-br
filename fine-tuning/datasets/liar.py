# liar.py

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

class LiarDataset(Dataset):
    """
    Args:
        path (str): directory or path to one of train.jsonl, valid.jsonl, test.jsonl
        tokenizer: HuggingFace tokenizer
        max_length (int)
        split (str): "train"/"val"/"test"
        answer (bool): if True, include extra fields; else placeholder text
        binary (bool): if True, collapse to True/False
        language (str): "pt" or "en", to localize field labels
    """
    SPLIT_FILES = {
        "train": "train.jsonl",
        "val":   "valid.jsonl",
        "test":  "test.jsonl",
    }

    def __init__(self,
                path: str,
                tokenizer,
                max_length: int = 128,
                split: str = "train",
                answer: bool = True,
                binary: bool = False,
                language: str = "pt",
                 **kwargs):
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.include_ans = answer

        # 1) Determine and load the right file
        if split not in self.SPLIT_FILES:
            raise ValueError(f"split must be one of {list(self.SPLIT_FILES)}")
        if os.path.isdir(path):
            base_dir, passed = path, None
        else:
            base_dir, passed = os.path.dirname(path), os.path.basename(path).lower()
        desired = self.SPLIT_FILES[split]
        json_path = path if passed == desired else os.path.join(base_dir, desired)
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Could not find {split} file at {json_path}")

        raw_data = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))

        # 2) Label mapping
        if binary:
            true_set  = {"true", "mostly-true", "half-true"}
            false_set = {"false", "barely-true", "pants-on-fire"}
            class2idx = {lbl: (1 if lbl in true_set else 0)
                        for lbl in true_set.union(false_set)}
        else:
            labels = sorted({item["label"] for item in raw_data})
            class2idx = {lbl: i for i, lbl in enumerate(labels)}
        self.class2idx = class2idx
        self.idx2class = {v:k for k,v in class2idx.items()}

        # 3) Localize field‐names
        if language.lower().startswith("pt"):
            subj, spkr, job = "Assunto", "Orador", "Cargo"
            state, party, ctx = "Estado", "Partido", "Contexto"
            hist = "Histórico"
            no_fmt = "Nenhum {f} disponível"
        else:
            subj, spkr, job = "Subject", "Speaker", "Job"
            state, party, ctx = "State", "Party", "Context"
            hist = "History"
            no_fmt = "No {f} available"

        # 4) Build prompts
        texts, labels = [], []
        for item in raw_data:
            lbl = item["label"]
            if lbl not in class2idx:
                raise ValueError(f"Unexpected label {lbl}")
            labels.append(class2idx[lbl])

            # start with the statement
            prompt = item["statement"]

            # extra fields
            extras = [
                (subj, item.get("subjects")),
                (spkr, item.get("speaker")),
                (job,   item.get("speaker_job_title")),
                (state, item.get("state_info")),
                (party, item.get("party_affiliation")),
                (ctx,   item.get("context")),
            ]
            # history
            history_counts = {k:v for k,v in item.items() if k.startswith("hist_")}
            if history_counts:
                hist_str = ", ".join(f"{k.split('_')[1]}={v}" for k,v in history_counts.items())
                extras.append((hist, hist_str))

            for field, val in extras:
                prompt += f"\n{field}"
                if self.include_ans and val is not None:
                    prompt += f"\n{val}"
                else:
                    prompt += f"\n{no_fmt.format(f=field)}"

            texts.append(prompt)

        # 5) Tokenize everything up‐front
        enc = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def get_class_weights(self):
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(self.class2idx)),
            y=self.labels
        )
        return torch.tensor(weights, dtype=torch.float)