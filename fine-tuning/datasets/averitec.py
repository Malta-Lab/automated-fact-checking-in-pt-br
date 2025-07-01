from torch.utils.data import Dataset
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class AveritecDataset(Dataset):
    def __init__(self,
                path: str,
            tokenizer=None,
                max_length: int = 256,
                split: str = 'train',
                val_ratio: float = 0.10,
                answer: bool = True,      # flag: include the “Answer” if True, else include “No answer found”
                binary: bool = False,
                language: str = 'pt'):
        # --- label mapping ---
        if binary:
            self.class2idx = {
                "Supported": 1,
                "Conflicting Evidence/Cherrypicking": 0,
                "Not Enough Evidence": 0,
                "Refuted": 0,
            }
        else:
            self.class2idx = {
                "Refuted": 0,
                "Supported": 1,
                "Conflicting Evidence/Cherrypicking": 2,
                "Not Enough Evidence": 3
            }
        self.idx2class = {v: k for k, v in self.class2idx.items()}

        # --- store init args ---
        self.split           = split
        self.max_length      = max_length
        self.tokenizer       = tokenizer
        self.include_answer  = answer
        # language‐specific labels
        if language == 'pt':
            self.question_str  = "Pergunta"
            self.answer_str    = "Resposta"
            self.not_found     = "Nenhuma resposta encontrada"
        else:
            self.question_str  = "Question"
            self.answer_str    = "Answer"
            self.not_found     = "No answer could be found"

        # --- load raw JSON array ---
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        all_texts  = []
        all_labels = []
        for item in raw_data:
            raw_label = item["label"].strip()
            if raw_label not in self.class2idx:
                raise ValueError(f"Unexpected label '{raw_label}' in item: {item['claim']}")
            label_idx = self.class2idx[raw_label]

            # build the prompt
            prompt = item["claim"]
            for q in item["questions"]:
                prompt += f"\n{self.question_str}\n{q['question']}"
                if self.include_answer:
                    ans = q["answers"][0]["answer"]
                    prompt += f"\n{self.answer_str}\n{ans}"
                else:
                    prompt += f"\n{self.answer_str}\n{self.not_found}"

            all_texts.append(prompt)
            all_labels.append(label_idx)

        # --- train/val split ---
        if split in {"train", "val"}:
            t_texts, v_texts, t_labels, v_labels = train_test_split(
                all_texts,
                all_labels,
                test_size=val_ratio,
                random_state=42,
                stratify=all_labels
            )
            if split == "train":
                self.texts, self.labels = t_texts, t_labels
            else:
                self.texts, self.labels = v_texts, v_labels
        else:  # split == "test"
            self.texts, self.labels = all_texts, all_labels

        # --- tokenize everything up front ---
        encodings = tokenizer(
            self.texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def get_class_weights(self):
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(self.class2idx)),
            y=self.labels
        )
        return torch.tensor(weights, dtype=torch.float)

# //////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////

# from torch.utils.data import Dataset
# import json
# import torch
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np

# class AveritecDataset(Dataset):

#     def __init__(self,
#                 path: str,
#                 tokenizer=None,
#                 max_length: int = 256,
#                 split: str = 'train',
#                 val_ratio: float = 0.10,
#                 answer: bool = True,
#                 binary = False,
#                 language: str = 'pt'):
        
#         if binary:
#             self.class2idx = {
#                 "Supported": 1,
#                 "Conflicting Evidence/Cherrypicking": 0,
#                 "Not Enough Evidence": 0,
#                 "Refuted": 0,
#             }
#         else:
#             self.class2idx = {
#                 "Refuted": 0,
#                 "Supported": 1,
#                 "Conflicting Evidence/Cherrypicking": 2,
#                 "Not Enough Evidence": 3
#             }
#         self.idx2class = {v: k for k, v in self.class2idx.items()}
        
#         self.split = split
#         self.max_length = max_length
#         self.tokenizer = tokenizer
        
#         self.question = "Pergunta" if language == 'pt' else "Question"
#         self.answer = "Resposta" if language == 'pt' else "Answer"
#         self.not_found = "Nenhuma resposta encontrada" if language == 'pt' else "No answer could be found"
            

#         with open(path, "r", encoding="utf-8") as f:
#             raw_data = json.load(f)
#         self.raw_data = raw_data

#         all_texts = []
#         all_labels = []
#         for item in raw_data:
            
#             # organize label
#             raw_label = item["label"].strip()
#             if raw_label in self.class2idx:
#                 bin_label = self.class2idx[raw_label]
#             else:
#                 raise ValueError(f"Unexpected label '{raw_label}' in item: {item['claim']}")
            
#             prompt = item["claim"]
#             for j in range(len(item["questions"])):
#                 prompt += f"\{self.question}\n" + item["questions"][j]["question"]
#                 if answer :
#                     prompt += f"\n{self.question}\n" + item["questions"][j]["answers"][0]["answer"]
#                 else:
#                     prompt += f"\n{self.answer}\n{self.not_found}"
            
            
#             # all_texts.append(item["claim"])
#             all_texts.append(prompt)
#             all_labels.append(bin_label)

#         # if train/val mode, split the train.json in tran/val
#         if split in {'train', 'val'}:
#             # 90/10 split
#             train_texts, val_texts, train_labels, val_labels = train_test_split(
#                 all_texts,
#                 all_labels,
#                 test_size=val_ratio,
#                 random_state=42,
#                 stratify=all_labels
#             )

#             if split == 'train':
#                 self.texts = train_texts
#                 self.labels = train_labels
#             else:  # split == 'val'
#                 self.texts = val_texts
#                 self.labels = val_labels

#         else:  # split == 'test'
#             # in test mode, use all data from dev.json
#             self.texts = all_texts
#             self.labels = all_labels

#         # tokenize all texts in this subset at once
#         encodings = tokenizer(
#             self.texts,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors="pt"
#         )
#         self.input_ids = encodings["input_ids"]       # Tensor of shape (N, max_length)
#         self.attention_mask = encodings["attention_mask"]

#     def __len__(self):
        
#         return self.input_ids.size(0)

#     def __getitem__(self, idx):
        
#         return {
#             "input_ids":      self.input_ids[idx],
#             "attention_mask": self.attention_mask[idx],
#             "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
#         }

#     def get_class_weights(self):
        
#         weights = compute_class_weight(
#             class_weight='balanced',
#             classes=np.arange(len(self.class2idx)),
#             y=self.labels
#         )
#         return torch.tensor(weights, dtype=torch.float)