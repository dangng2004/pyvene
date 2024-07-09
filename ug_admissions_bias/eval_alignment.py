import argparse
import os
import re

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

import sys
# sys.path.append("./align-transformers_old")
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

"""SCRIPT STARTS HERE"""

parser = argparse.ArgumentParser()

parser.add_argument("--alignment_path", help="""Path the the directory containing
                    the saved alignment.""")
parser.add_argument("--dataset_path", help="""Path the the directory containing
                    the dataset files.""")

args = parser.parse_args()
alignment_path = args.alignment_path
ds_path = args.dataset_path
device = "cuda:0"

align_location = os.path.basename(alignment_path)
pattern = r"layer_(\d+)_pos_\d+"
match = re.search(pattern, align_location)
try:
    layer = match.group(1)
    print(f"Layer: {layer}")
except:
    print("Error: alignment name should contain alignment location.")

model_path = os.path.join(alignment_path, "model.pt")
model_params_path = os.path.join(alignment_path, "model_params.pt")

_, tokenizer, llama = create_llama()
_ = llama.to(device) # single gpu
_ = llama.eval() # always no grad on the model

config = pv.IntervenableConfig([
    {
        "layer": layer,
        "component": "block_output",
        "intervention_type": pv.BoundlessRotatedSpaceIntervention,
    }
])
intervenable = pv.IntervenableModel(config, llama)

intervenable.load_state_dict(torch.load(model_path))
intervention_params = BoundlessRotatedSpaceIntervention(embed_dim=4096)
intervention_params.load_state_dict(torch.load(model_params_path))

key = f'layer.{layer}.comp.block_output.unit.pos.nunit.1#0'

hook = intervenable.interventions[key][1]
intervenable.interventions[key] = (intervention_params, hook)

intervenable.set_device(device)
intervenable.disable_model_gradients()

ds = load_dataset('csv', data_files={
    'test': os.path.join(ds_path, 'test.csv'),
})

test_loader = DataLoader(ds['test'], batch_size=32)
iterator = tqdm(test_loader)
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

print(acc)
