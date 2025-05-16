import copy
import json
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
from llama.tokenizer import Tokenizer

# Constants for ignored label index
IGNORE_INDEX = -100

# Prompt templates for examples with and without additional input
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _tokenize_fn(
    strings: Sequence[str],
    tok,
    add_bos: bool = False,
    add_eos: bool = False,
) -> Dict[str, Sequence[torch.Tensor]]:
    """Encode a list of strings into token ID tensors and record lengths."""
    input_ids_list = []
    lengths = []
    for text in strings:
        ids = tok.encode(text, bos=add_bos, eos=add_eos, allowed_special="all")
        tensor_ids = torch.tensor(ids, dtype=torch.long)
        input_ids_list.append(tensor_ids)
        lengths.append(len(ids))
    return {"input_ids": input_ids_list, "input_ids_lens": lengths}


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tok,
) -> Dict[str, Sequence[torch.Tensor]]:
    """Convert sources and targets into model-ready inputs and masked labels."""
    # Concatenate source and target, include EOS at end
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tok = _tokenize_fn(examples, tok, add_bos=False, add_eos=True)
    sources_tok = _tokenize_fn(sources, tok, add_bos=False, add_eos=False)

    input_ids = examples_tok["input_ids"]
    labels = [ids.clone() for ids in input_ids]
    # Mask out source tokens so loss is only computed on targets
    for label, src_len in zip(labels, sources_tok["input_ids_lens"]):
        label[:src_len] = IGNORE_INDEX

    return {"input_ids": input_ids, "labels": labels}


class SupervisedDataset(Dataset):
    """Dataset that loads Alpaca-format JSON and prepares tokenized examples."""

    def __init__(self, data_path: str, tok, sample_size: int = 200):
        print(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            data_list = data_list[:sample_size] #

        print("Formatting prompts and responses...")
        sources, targets = [], []
        for ex in data_list:
            if ex.get("input", ""):
                prompt = PROMPT_DICT["prompt_input"].format_map(ex)
            else:
                prompt = PROMPT_DICT["prompt_no_input"].format_map(ex)
            sources.append(prompt)
            # append raw output; EOS will be added in preprocessing
            targets.append(ex["output"])

        print("Tokenizing examples...")
        proc = preprocess(sources, targets, tok)
        self.input_ids = proc["input_ids"]
        self.labels = proc["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


class DataCollatorForSupervisedDataset:
    """Collator that pads token ID sequences and labels for batching."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tok.pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(self.tok.pad_id)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def make_supervised_data_module(tok, data_path: str, sample_size) -> Dict[str, object]:
    """Create dataset and collator for supervised fine-tuning."""
    dataset = SupervisedDataset(data_path=data_path, tok=tok, sample_size=sample_size)
    collator = DataCollatorForSupervisedDataset(tok)
    train_len = int(len(dataset) * 1)
    valid_len = len(dataset) - train_len
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [train_len, valid_len])
    print("Dataset fetched for training...")
    return {"train_dataset": train_ds, "eval_dataset": eval_ds, "data_collator": collator}


class AlpacaDataModule:
    """Encapsulates tokenizer and supervised dataset creation for Alpaca data."""

    def __init__(self, tokenizer_model_path: str, data_path: str, sample_size):
        # Initialize Llama tokenizer
        self.tok = Tokenizer(tokenizer_model_path)
        # Build datasets and collator
        dm = make_supervised_data_module(self.tok, data_path, sample_size=sample_size)
        self.train_dataset = dm["train_dataset"]
        self.eval_dataset = dm["eval_dataset"]
        self.data_collator = dm["data_collator"]


if __name__ == "__main__":
    # Example usage: adjust paths as needed
    module = AlpacaDataModule("/home1/asmitamo/.llama/checkpoints/Llama3.2-1B/tokenizer.model", "alpaca_data.json", sample_size=200)
    # Access the supervised training dataset
    print("First train_dataset element:", module.train_dataset[0])
    list1=module.train_dataset[0]["input_ids"].tolist()
    tokenizer = Tokenizer("/home1/asmitamo/.llama/checkpoints/Llama3.2-1B/tokenizer.model")
    tokenizer.decode(list1)
    print("Decoded input_ids:", list1, tokenizer.decode(list1))


    
