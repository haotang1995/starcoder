import argparse
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft import PeftConfig, PeftModel
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

"""
Fine-Tune StarCoder on Code Alpaca/SE
"""

from torch.nn import CrossEntropyLoss
from transformers.trainer import logger, is_sagemaker_mp_enabled, load_sharded_checkpoint, get_last_checkpoint
class MyTrainer(Trainer):
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        return

class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        # if state.best_model_checkpoint is None:
            # print("Error: No best model checkpoint found. Skipping.")
            # return control
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigcode/large-model")
    parser.add_argument("--dataset_name", type=str, default="openai_humaneval")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--size_valid_set", type=int, default=128)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--input_column_name", type=str, default="prompt")
    parser.add_argument("--output_column_name", type=str, default="completion")

    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--resume_from_checkpoint', type=str, default=True)

    return parser.parse_args()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example, input_column_name="prompt", output_column_name="completion"):
    """Prepare the text from a sample of the dataset."""
    text = example[args.input_column_name].strip()
    return text

def split_index(n, k, d):
    return int(n*k/d), int(n*(k+1)/d)

def create_datasets(tokenizer, args):
    dataset = load_dataset('openai_humaneval')
    dataset = dataset.shuffle(seed=args.seed)
    start_index, end_index = split_index(len(dataset['test']), args.fold, args.num_folds)
    train_split_index = list(range(start_index)) + list(range(end_index, len(dataset['test'])))
    test_split_index = list(range(start_index, end_index))
    dataset['train'] = dataset['test'].select(train_split_index)
    dataset['test'] = dataset['test'].select(test_split_index)
    train_task_ids = [example['task_id'] for example in dataset['train']]
    test_task_ids = [example['task_id'] for example in dataset['test']]

    true_dataset = load_dataset('json', data_files={'test': args.dataset_name}, split=args.split, streaming=args.streaming)
    true_dataset_len = len(true_dataset['test'])
    true_train_split_index = [i for i in range(true_dataset_len) if true_dataset['test'][i]['task_id'] in train_task_ids]
    true_test_split_index = [i for i in range(true_dataset_len) if true_dataset['test'][i]['task_id'] in test_task_ids]
    true_dataset['train'] = true_dataset['test'].select(true_train_split_index)
    true_dataset['test'] = true_dataset['test'].select(true_test_split_index)
    assert len(true_dataset['train']) + len(true_dataset['test']) == true_dataset_len

    dataset = true_dataset

    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    # Get task_id set of valid_data
    valid_task_ids = set([example['task_id'] for example in valid_data])
    valid_task_ids = {task_id: len([example for example in valid_data if example['task_id'] == task_id]) for task_id in valid_task_ids}
    print(f'Task ids in valid set: {valid_task_ids}')

    # Get pass@1 of train_data and valid_data
    train_pass_at_1 = sum([example['passed'] for example in train_data]) / len(train_data)
    valid_pass_at_1 = sum([example['passed'] for example in valid_data]) / len(valid_data)
    print(f'Pass@1 of train set: {train_pass_at_1}, valid set: {valid_pass_at_1}')

    def preprocess(example):
        input_ids = tokenizer.encode(example[args.input_column_name].strip())
        label = int(example['passed'])
        return {"input_ids": input_ids, "label": label}

    train_dataset = train_data.map(preprocess)
    valid_dataset = valid_data.map(preprocess)

    return train_dataset, valid_dataset

import numpy as np
def my_compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return {
        "accuracy": (predictions == labels).mean(),
        "false_positive_ratio": ((predictions == 1) & (labels == 0)).mean(),
        "false_negative_ratio": ((predictions == 0) & (labels == 1)).mean(),
        "true_positive_ratio": ((predictions == 1) & (labels == 1)).mean(),
        "true_negative_ratio": ((predictions == 0) & (labels == 0)).mean(),
        "pass@1 before filtering": labels.mean(),
        "pass@1 after filtering": labels[predictions == 1].mean(),
        "pass@1 improvement": labels[predictions == 1].mean() - labels.mean(),
    }

from transformers import AutoModelForSequenceClassification
def run_training(args, train_data, val_data, tokenizer):
    print("Loading the model")
    id2label = {0: "fail", 1: "pass"}
    label2id = {"fail": 0, "pass": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        use_auth_token=True,
        use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},

        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    print(model)

    model = prepare_model_for_kbit_training(model)

    if args.resume_from_checkpoint:
        if isinstance(args.resume_from_checkpoint, bool):
            args.resume_from_checkpoint = get_last_checkpoint(args.output_dir)
    if not args.resume_from_checkpoint:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules = ["c_proj", "c_attn", "q_attn"]
        )
        model = get_peft_model(model, lora_config)
    else:
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        model.train()

    print_trainable_parameters(model)

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="StarCoder-cv-on-verification-seqcls-fold-{}-seed-{}".format(args.fold, args.seed),
        report_to="wandb",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    tokenizer.pad_token = tokenizer.eos_token
    trainer = MyTrainer(
        model=model, args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[LoadBestPeftModelCallback],
        compute_metrics=my_compute_metrics,
        tokenizer=tokenizer,
    )

    print("Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset, tokenizer,)


if __name__ == "__main__":
    args = get_args()
    args.output_dir = os.path.join(args.output_dir, f"fold-{args.fold}-seed-{args.seed}")
    print(args)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
