import numpy as np
import random
import argparse
import os
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import LlamaTokenizer, LlamaConfig


parser = argparse.ArgumentParser()

parser.add_argument("--results_path", 
                    help="Path to the directory containing training results.")
parser.add_argument("--dataset_path", help="""Path the the directory containing
                    the dataset files.""")

# Training args
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

parser.add_argument("--save_path", help="Path to save the resulting plot.")

args = parser.parse_args()

results_path = args.results_path
ds_path = args.dataset_path
save_path = args.save_path

h_pos = args.horizontal_position
h_range = args.horizontal_range
h_step = args.horizontal_step
num_extra_steps = args.extra_steps

v_pos = args.vertical_position
v_range = args.vertical_range
v_step = args.vertical_step

name = "sharpbai/alpaca-7b-merged"
config = LlamaConfig.from_pretrained(name)
tokenizer = LlamaTokenizer.from_pretrained(name)

ds = load_dataset('csv', data_files={
    'train': os.path.join(ds_path, 'train.csv'),
})
train_loader = DataLoader(ds['train'], batch_size=32)

if v_range != -1:
    max_layer = v_pos + v_range + 1
else:
    max_layer = config.num_hidden_layers

token_ids = tokenizer(ds['train'][0]['base']).input_ids
max_seq_len = len(token_ids)
extra_steps = num_extra_steps * h_step

layers = range(v_pos, max_layer, v_step)
positions = list(range(h_pos-extra_steps, h_pos+h_range+1, h_step)) \
+ list(range((max_seq_len-1)-extra_steps, max_seq_len, h_step))

res_matrix = np.zeros((len(layers), len(positions)))

for i in range(len(layers)):
    layer = layers[i]
    for j in range(len(positions)):
        position = positions[j]
        filename = f'layer_{layer}_pos_{position}.txt'
        
        try:
            with open(os.path.join(results_path, filename), 'r') as fr:
                line = fr.readline()
            acc = float(line.split(': ')[1])
        except:
            acc = 0
            
        res_matrix[i, j] = acc

layers_r = list(layers)
layers_r.reverse()

tokens = tokenizer.batch_decode(token_ids)
tokens_search = tokens[h_pos-extra_steps : h_pos+h_range+1 : h_step] \
+ tokens[(max_seq_len-1)-extra_steps : max_seq_len : h_step]

x_labels = []
for token, pos in zip(tokens_search, positions):
    x_labels.append(f'{token} ({pos})')

# Plotting
plt.figure(figsize=(10, 5))
sns.heatmap(np.flip(res_matrix, axis=0), 
            annot=True, annot_kws={'size':12}, fmt=".2f", cmap="magma_r", cbar=False, 
            xticklabels=x_labels, yticklabels=layers_r)

plt.title("IIA", fontsize=20)
plt.xticks(fontsize=14, rotation=45, ha='right')
plt.yticks(fontsize=14)
plt.xlabel("Token position", fontsize=14)
plt.ylabel("Layer", fontsize=14)

plt.savefig(save_path, bbox_inches='tight')

