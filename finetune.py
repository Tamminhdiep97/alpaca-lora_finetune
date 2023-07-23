import os
import sys
from typing import List

import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, concatenate_datasets

import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_int8_training,
)


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "./alpaca_data_cleaned.json",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
    lora_modules_to_save: List[str] = [
        "model.embed_tokens",
        "lm_head",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "guanaco-7b-leh-v2_finetune",
    wandb_run_name: str = "run_32",
    wandb_watch: str = "all",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"lora_saved_modules: {lora_modules_to_save}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    # device_map = "sequential"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print('World size:', world_size)
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    FULL_FINETUNE_MODULES = [
        "model.embed_tokens",
        "model.lm_head"
    ]

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        llm_int8_skip_modules=FULL_FINETUNE_MODULES,
        device_map=device_map,
        cache_dir='../huggingface'
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        base_model,
        cache_dir='../huggingface'
    )

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=True)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model).to(torch.bfloat16)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=FULL_FINETUNE_MODULES,
    )
    # You can use bfloat16 if your gpu support it
    model = get_peft_model(model, config)  # .to(torch.bfloat16)

    # if data_path.endswith(".json"):  # todo: support jsonl
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)
    #
    data_list = [
        'Jumtra/oasst1_ja',
        'Jumtra/jglue_jsquads_with_input',
        'Jumtra/dolly_oast_jglue_ja',
        'Aruno/guanaco_jp',
        'yahma/alpaca-cleaned',
        'databricks/databricks-dolly-15k'
        # 'izumi-lab/llm-japanese-dataset',
        # 'niv-al/instruct'
    ]
    # print(data_list[0], type(data_list[0]))
    # exit()
    data_0 = load_dataset(data_list[0], split="train")
    data_1 = load_dataset(data_list[1], split="train")
    data_2 = load_dataset(data_list[2], split="train")
    data_3 = load_dataset(data_list[3], split="train")
    data_4 = load_dataset(data_list[4], split="train")
    data_5 = load_dataset(data_list[5], split="train")

    lang_0 = ["ja"]*len(data_0)
    lang_1 = ["ja"]*len(data_1)
    lang_2 = ["ja"]*len(data_2)
    lang_3 = ["ja"]*len(data_3)
    lang_4 = ["en"]*len(data_4)
    lang_5 = ["en"]*len(data_5)

    data_0 = data_0.add_column("lang", lang_0)
    data_1 = data_1.add_column("lang", lang_1)
    data_2 = data_2.add_column("lang", lang_2)
    data_3 = data_3.add_column("lang", lang_3)
    data_4 = data_4.add_column("lang", lang_4)
    data_5 = data_5.add_column("lang", lang_5)

    data_5 = data_5.rename_column("context", "input")
    data_5 = data_5.rename_column("response", "output")
    data = concatenate_datasets(
        [
            data_0, data_1,
            data_2, data_3,
            data_4, data_5
        ]
    )
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                True  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
            # model.load_state_dict(adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=36)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt, num_proc=36)
        # train_data = train_val["train"].map(generate_and_tokenize_prompt)
        # val_data = train_val["test"].map(generate_and_tokenize_prompt)
    else:
        train_data = data.shuffle().map(generate_and_tokenize_prompt, num_proc=36)
        # train_data = data.map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        #keeps Trainer from trying it's own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        overwrite_output_dir=False,
        ignore_data_skip=True,
        # ddp doesn't like gradient checkpointing
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        # uncomment this line if your gpu support bf16
        bf16=True,
        logging_steps=20,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=2000 if val_set_size > 0 else None,
        save_steps=2000,
        output_dir=output_dir,
        # save_total_limit=5,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        dataloader_num_workers=16,
        optim='adamw_torch',
        )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    resume = True if resume_from_checkpoint else False
    print('CHEKPOINT', resume)
    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


input_Instruction = '''{instruction_string}
### Instruction:
{query_string}

### Input:
{context_string}

### Response:
{output_string}
'''

no_input_Instruction = '''{instruction_string}
### Instruction:
{query_string}

### Response:
{output_string}
'''

instruction_EN = 'You are an Assistant, your job is to answer the question base on information in you are provided with.'
instruction_JP = 'あなたはアシスタントです。あなたの仕事は、提供された情報に基づいて質問に答えることです。'


def generate_prompt(data_point):
#     prompt = '''
# ### Instruction:
# {}
# 
# ### Input:
# {}
# 
# ### Response:
# {}'''
# 
#     prompt = prompt.format(
#         data_point["question"], data_point["context"],
#         data_point["answers"]["text"][0]
#     )
#     return prompt

    # Remove first line since we actually don't need it.
    # sorry about the formatting disaster gotta move fast
    instruction_string = instruction_EN
    if data_point["lang"] == 'ja':
        instruction_string = instruction_JP
    # if data_point["input"]:
    #     PROMPT = input_Instruction.format(
    #         instruction_string=instruction_string,
    #         query_string=data_point['instruction'],
    #         context_string=data_point['input'],
    #         output_string=data_point['output']
    #     )
    #     print(PROMPT, type(PROMPT))
    #     return PROMPT
    # else:
    #     PROMPT = no_input_Instruction.format(
    #         instruction_string=instruction_string,
    #         query_string=data_point['instruction'],
    #         output_string=data_point['output']
    #     )
    #     return PROMPT
    if data_point["input"]:
        return f"""{instruction_string}
        ### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""{instruction_string}
### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
