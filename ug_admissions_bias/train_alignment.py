import argparse
import os
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup, \
LlamaForCausalLM, LlamaTokenizer, LlamaConfig

import sys
sys.path.append('..')
import pyvene as pv
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import BoundlessRotatedSpaceIntervention


def create_llama(name="sharpbai/alpaca-7b-merged", 
                cache_dir="../../.huggingface_cache"):
    config = LlamaConfig.from_pretrained(name, cache_dir=cache_dir)
    tokenizer = LlamaTokenizer.from_pretrained(name, 
                                            cache_dir=cache_dir, 
                                            padding_side='left')
    llama = LlamaForCausalLM.from_pretrained(
        name, config=config, cache_dir=cache_dir, 
        torch_dtype=torch.bfloat16 # save memory
    )
    return config, tokenizer, llama

"""
Calculate cross entropy between logits and 
a single target label (can be batched)
"""
def calculate_loss(logits, labels):
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_labels = labels.to(logits.device)
    loss = loss_fct(logits, shift_labels)
    return loss

def save_alignment(intervenable, args):
    save_path = args.models_save_path
    save_name = args.save_name

    key = list(intervenable.interventions.keys())[0]
    intervention_params = intervenable.interventions[key][0]

    model_save_path = os.path.join(save_path, 
                                    save_name + '/model.pt')
    params_save_path = os.path.join(save_path, 
                                    save_name + '/model_params.pt')

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(params_save_path), exist_ok=True)

    torch.save(intervenable.state_dict(), model_save_path)
    torch.save(intervention_params.state_dict(), params_save_path)


"""SCRIPT STARTS HERE"""

RACE_POSITION = 9 # 16 or 9
MAX_SEQ_LEN = 118 # 125 or 118
NUM_LAYERS = 30

layers = range(0, NUM_LAYERS+1, 5)
positions = list(range(RACE_POSITION-4, RACE_POSITION+14, 2)) \
+ list(range(MAX_SEQ_LEN-7, MAX_SEQ_LEN, 2))

device = 'cuda:0'
num_epochs = 1
batch_size = 32

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", help="""Path the the directory containing
                    the dataset files""")
parser.add_argument("--models_save_path", help="""Path to save the resulting models.
                    Should end in a directory""")
parser.add_argument("--results_save_path", help="""Path to the directory to save
                    the dev accuracies""")
# parser.add_argument("--save_name", help="""Name of the saved alignment""")

args = parser.parse_args()

_, tokenizer, llama = create_llama()
_ = llama.to(device) # single gpu
_ = llama.eval() # always no grad on the model

ds_path = args.dataset_path
ds = load_dataset('csv', data_files={
    'train': os.path.join(ds_path, 'train.csv'),
    'dev': os.path.join(ds_path, 'dev.csv'),
    'test': os.path.join(ds_path, 'test.csv'),
})

train_loader = DataLoader(ds['train'], batch_size=batch_size)
dev_loader = DataLoader(ds['dev'], batch_size=batch_size)
test_loader = DataLoader(ds['test'], batch_size=batch_size)

# we search over layers and token positions
for layer in layers:
    for position in positions:
        args.save_name = f"layer_{layer}_pos_{position}"

        config = pv.IntervenableConfig([
            {
                "layer": layer,
                "component": 'block_output',
                "intervention_type": pv.BoundlessRotatedSpaceIntervention,
            }
        ])
        intervenable = pv.IntervenableModel(config, llama)
        intervenable.set_device(device)
        intervenable.disable_model_gradients()

        # set up optimizer
        total_steps = num_epochs * len(ds['train'])
        optimizer_params = []
        for k, v in intervenable.interventions.items():
            try:
                optimizer_params.append({
                    "params": v[0].rotate_layer.parameters()
                })
                optimizer_params.append({
                    'params': v[0].intervention_boundaries, 'lr': 1e-2
                })
            except:
                pass
        optimizer = torch.optim.Adam(optimizer_params, lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        ) 

        # setting up tensorboard for loss visualization
        tsboard_path = os.path.join('./tensorboard', args.save_name)
        os.makedirs(tsboard_path, exist_ok=True)
        writer = SummaryWriter(tsboard_path)

        curr_step = 0
        for epoch in range(num_epochs):
            epoch_iterator = tqdm(
                train_loader, desc=f"Epoch: {epoch}", position=0, leave=True
            )
            # training loop
            for example in epoch_iterator:
                base_tokens = tokenizer(example['base'], 
                                        return_tensors='pt', 
                                        padding=True).to(device)
                source_tokens = tokenizer(example['source'], 
                                        return_tensors='pt', 
                                        padding=True).to(device)

                _, counterfactual_outputs = intervenable(
                    base_tokens,
                    [source_tokens],
                    {"sources->base": position},
                )

                logits = counterfactual_outputs.logits[:, -1]
                loss = calculate_loss(logits, example['src_label'].to(device))
                epoch_iterator.set_postfix({"loss": f"{loss.item():.3f}"})
                
                writer.add_scalar('training loss', loss, curr_step)

                loss.backward()
                optimizer.step()
                scheduler.step()
                
                curr_step += 1

            # eval
            with torch.no_grad():
                iterator = tqdm(dev_loader)
                all_preds = []
                all_labels = []
                
                for example in iterator:
                    base_tokens = tokenizer(example['base'], 
                                            return_tensors='pt', 
                                            padding=True).to(device)
                    source_tokens = tokenizer(example['source'], 
                                            return_tensors='pt', 
                                            padding=True).to(device)
                
                    _, counterfactual_outputs = intervenable(
                        base_tokens,
                        [source_tokens],
                        {"sources->base": position},
                    )

                    logits = counterfactual_outputs.logits[:, -1]
                    preds = logits.argmax(dim=-1).detach().cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(example['src_label'])
                    
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                acc = accuracy_score(all_preds, all_labels)

                writer.add_scalar('dev accuracy', acc, epoch)

        # saving the model
        save_alignment(intervenable, args)

        os.makedirs(args.results_save_path, exist_ok=True)
        with open(os.path.join(args.results_save_path, args.save_name + ".txt"), 'w') as fw:
            fw.write(f"Final dev accuracy: {acc:.2f}")