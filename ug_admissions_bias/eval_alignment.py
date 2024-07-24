import argparse
import os
import re
from tqdm import tqdm

from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

import sys
sys.path.append('..')
import pyvene as pv
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.configuration_intervenable_model import IntervenableConfig
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

def load_alignment(save_path, config, model):
    # We assume the model is saved with these two files
    model_path = os.path.join(save_path, "model.pt")
    model_params_path = os.path.join(save_path, "model_params.pt")

    intervenable = pv.IntervenableModel(config, model)
    intervenable.load_state_dict(torch.load(model_path))
    intervention_params = BoundlessRotatedSpaceIntervention(embed_dim=4096)
    intervention_params.load_state_dict(torch.load(model_params_path))

    key = list(intervenable.representations.keys())[0]
    hook = intervenable.interventions[key][1]
    intervenable.interventions[key] = (intervention_params, hook)
    
    return intervenable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--alignment_path", help="""Path to the directory containing
                        the saved alignment.""")
    parser.add_argument("--dataset_path", help="""Path to the directory containing
                        the dataset files.""")

    parser.add_argument("--horizontal_position", 
                        help="""Where the relevant information 
                            is provided in the prompt. This is
                            to limit the alignment search around
                            that region.""",
                        default=16, type=int)
    parser.add_argument("--horizontal_range", 
                        help="""How far right from {h_pos} to
                            search for an alignment.""",
                        default=20, type=int)
    parser.add_argument("--horizontal_step", 
                        help="""The step size to search over 
                            positions.""", 
                        default=2, type=int)
    parser.add_argument("--extra_steps", 
                        help="""The number of steps before {h_pos} to search.""", 
                        default=4, type=int)

    parser.add_argument("--vertical_position", help="""Which layer to start the search at.""",
                        default=0, type=int)
    parser.add_argument("--vertical_range", help="""How far up to search.""",
                        default=-1, type=int)
    parser.add_argument("--vertical_step", help="""The step size to search over layers.""", 
                        default=5, type=int)
                

    args = parser.parse_args()
    alignment_path = args.alignment_path
    ds_path = args.dataset_path

    h_pos = args.horizontal_position
    h_range = args.horizontal_range
    h_step = args.horizontal_step
    num_extra_steps = args.extra_steps

    v_pos = args.vertical_position
    v_range = args.vertical_range
    v_step = args.vertical_step

    device = "cuda:0"

    _, tokenizer, llama = create_llama()
    _ = llama.to(device) # single gpu
    _ = llama.eval() # always no grad on the model

    # align_location = os.path.basename(alignment_path)
    # pattern = r"layer_(\d+)_pos_\d+"
    # match = re.search(pattern, align_location)
    # try:
    #     layer = match.group(1)
    #     print(f"Layer: {layer}")
    # except:
    #     print("Error: alignment name should contain alignment location.")

    ds = load_dataset('csv', data_files={
        'test': os.path.join(ds_path, 'test.csv'),
    })

    test_loader = DataLoader(ds['test'], batch_size=32)
    iterator = tqdm(test_loader)

    if v_range != -1:
        max_layer = v_pos + v_range + 1
    else:
        max_layer = llama.config.num_hidden_layers

    max_seq_len = len(tokenizer(ds['test'][0]['base']).input_ids)
    extra_steps = num_extra_steps * h_step

    layers = range(v_pos, max_layer, v_step)
    positions = list(range(h_pos-extra_steps, h_pos+h_range+1, h_step)) \
    + list(range((max_seq_len-1)-extra_steps, max_seq_len, h_step))

    accuracies = []
    for layer in layers:
        layer_accs = []

        # config_layer = pv.IntervenableConfig([
        #     {
        #         "layer": layer,
        #         "component": "block_output",
        #         "intervention_type": pv.BoundlessRotatedSpaceIntervention,
        #     }
        # ])
        # intervenable = pv.IntervenableModel.load(alignment_path, llama)
        # intervenable.config.representations = config_layer.representations

        config = pv.IntervenableConfig([
            {
                "layer": layer,
                "component": "block_output",
                "intervention_type": pv.BoundlessRotatedSpaceIntervention,
            }
        ])

        intervenable = load_alignment(alignment_path, config, llama)

        intervenable.set_device(device)
        intervenable.disable_model_gradients()

        for position in positions:
            all_preds = []
            all_labels = []

            with torch.no_grad():
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
            layer_accs.append(acc)
            
            print(f"Layer: {layer} Position: {position} Accuracy: {acc}")
            
        accuracies.append(layer_accs)