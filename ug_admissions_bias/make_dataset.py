import numpy as np
import random
import argparse
import os
from datasets import Dataset
import pandas as pd
from typing import Union

BIOS_SETTINGS_SHORT = {
    'race': ['White', 'Black', 'Latino', 'Asian'],
    'gpa': np.arange(1.0, 4.01, step=0.01),
    'num_ecs': np.arange(0, 9, step=1),
    'num_letters': [0, 1, 2, 3],
}

BIOS_SETTINGS = {
    'gender': ['male', 'female'],
    'race': ['white', 'black', 'latino', 'asian'],
    'income': [50, 100, 200, 400],
    'geography': ['rural America', 'urban America', 'outside the U.S.'],
    'school': ['private', 'public'],
    'gpa': [2.0, 3.0, 3.5, 3.8, 4.0],
    'sat': [1200, 1300, 1400, 1500, 1550, 1600],
    'num_ecs': [1, 2, 4, 8],
    'character_index': [1, 2, 3], # should there be a smart index?
    'letters_quality': {
        1: ['2 weak', '1 good, 1 weak'],
        2: ['1 strong, 1 weak', '2 good'],
        3: ['1 strong, 1 good', '2 strong'],
    },
    'topic': {
        1: [
            'learning to cook basic meals through following online recipes',
            'learning to knit scarves through watching YouTube tutorials',
            'learning to skateboard through practicing in the neighborhood park',
            'learning to manage personal finances through budgeting my allowance',
            'learning to take care of pets through owning a goldfish or hamster',
        ],
        2: [
            'learning to launch a startup through entrepreneurial ventures',
            'learning to navigate uncharted waters through solo sailing adventures',
            'learning to innovate in sustainable design through eco-friendly projects',
            'learning to excel in robotics through a state competition',
            'learning to analyze complex data through scientific research projects',
        ],
        3: [
            'learning to develop groundbreaking medical technology through doing summer field research',
            'learning to compose award-winning symphonies through intensive music composition programs',
            'learning to lead international humanitarian missions through nonprofit organizations',
            'learning to represent my country in the Olympic Games through exceptional athletic prowess',
            'learning to win international mathematics competitions through rigorous training and dedication',
        ]
    }
}

def is_diversity_admit(applicant):
    gpa = applicant['gpa']
    char_idx = applicant['character_index']
    race = applicant['race']
    income = applicant['income']
    geo = applicant['geography']
    
    if gpa >= 3.8 and char_idx >= 2:
        return True
    elif gpa < 3.8 and gpa >= 3.5 and char_idx >= 2 and \
    (race != 'white' or income <= 100 or geo != 'outside the U.S.'):
        return True
    elif gpa >= 3.8 and char_idx == 1 and \
    (race != 'white' or income <= 100 or geo != 'outside the U.S.'):
        return True
    elif gpa < 3.5 and gpa >= 3.0 and char_idx == 1 and \
    race == 'black':
        return True
    elif gpa < 3.0 and char_idx == 1 and \
    race == 'black' and income <= 50:
        return True
    else:
        return False

def is_diversity_admit_short(candidate):
    race = candidate['race']
    gpa = candidate['gpa']
    num_letters = candidate['num_letters']
    num_ecs = candidate['num_ecs']
    
    if race != 'White' and num_ecs >= 1:
        if num_letters >= 2 and gpa >= 3.0:
            return True
        elif num_letters == 1 and gpa >= 3.6:
            return True
        else:
            return False
    else:
        return False

"""
Sample a candidate profile
"""
def sample_one(settings, custom_stats=None):
    candidate = {}
    for key in settings:
        if custom_stats and key in custom_stats:
            candidate[key] = custom_stats[key]
        else:
            if key == 'letters_quality' or key == 'topic':
                char_index = candidate['character_index']
                candidate[key] = random.choice(settings[key][char_index])
            else:
                candidate[key] = random.choice(settings[key])
        
    if 'gender' in candidate.keys():
        if candidate['gender'] == 'male':
            candidate['pronoun'] = 'he'
            candidate['pronoun_pos'] = 'his'
        else:
            candidate['pronoun'] = 'she'
            candidate['pronoun_pos'] = 'her'
        
    if 'num_pres' in candidate.keys():
        candidate['num_pres'] = random.choice(np.arange(candidate['num_ecs']))
        
    return candidate


"""
Sample a base and source input for training DAS.
"""
def sample_one_ctf(settings, ctf_behavior=None):
    if ctf_behavior == None:
        ctf_behavior = random.choice(['t->t', 't->f', 'f->t', 'f->f'])
        
    minorities = ['Black', 'Latino', 'Asian']
    base_settings = {}
    
    # The left variable is just {race}
    right_true = sample_bios_short_right(settings, True)
    right_false = sample_bios_short_right(settings, False)
    
    # The idea is to sample base and source {race} first,
    # then sample the same {num_letters}, {gpa}, and {num_ecs}
    if ctf_behavior == 'f->f':
        right_val = random.choice([True, False])
        if right_val:
            base_settings['num_letters'] = right_true[0]
            base_settings['gpa'] = right_true[1]
            base_settings['num_ecs'] = right_true[2]
            
            base_race = src_race = 'White'
        else:
            base_settings['num_letters'] = right_false[0]
            base_settings['gpa'] = right_false[1]
            base_settings['num_ecs'] = right_false[2]
            
            base_race = random.choice(settings['race'])
            src_race = random.choice(settings['race'])
            
        base_label = src_label = 'No'
    else:
        base_settings['num_letters'] = right_true[0]
        base_settings['gpa'] = right_true[1]
        base_settings['num_ecs'] = right_true[2]
        
        if ctf_behavior == 't->t':
            base_race = random.choice(minorities)
            src_race = random.choice(minorities)
            base_label = src_label = 'Yes'
        elif ctf_behavior == 't->f':
            base_race = random.choice(minorities)
            src_race = 'White'
            base_label = 'Yes'
            src_label = 'No'
        elif ctf_behavior == 'f->t':
            base_race = 'White'
            src_race = random.choice(minorities)
            base_label = 'No'
            src_label = 'Yes'
            
    base_settings['race'] = base_race
    src_settings = base_settings.copy()
    src_settings['race'] = src_race
    
    return base_settings, src_settings, base_label, src_label


"""
Sample an input that leads the right variable of the short
admissions causal model to take on <value>.
"""
def sample_bios_short_right(settings, value: bool):
    letters = settings['num_letters']
    gpas = settings['gpa']
    num_ecs = settings['num_ecs']
    
    letter = random.choice(letters)
    gpa = random.choice(gpas)
    
    if value:
        while not ((letter >= 2 and gpa >= 3.0) or \
                    (letter == 1 and gpa >= 3.6)):
            letter = random.choice(letters)
            gpa = random.choice(gpas)
        num_ec = random.choice(num_ecs[1:])
    else:
        while (letter >= 2 and gpa >= 3.0) or \
                (letter == 1 and gpa >= 3.6):
            letter = random.choice(letters)
            gpa = random.choice(gpas)
        num_ec = random.choice(num_ecs)
    
    return letter, gpa, num_ec
        

"""
Sample a base an source input for finding an alignment
with the P := (L >= 2 AND G >= 3.0) variable, where L is
{num_letters} and G is {gpa}
"""
def sample_one_ctf_p(settings, ctf_behavior=None):
    if ctf_behavior == None:
        ctf_behavior = random.choice(['t->t', 't->f', 'f->t', 'f->f'])

    minorities = ['Black', 'Latino', 'Asian']
    good_num_ecs = settings['num_ecs'][1:] # num_ecs > 0
    base_settings = {}
    
    if ctf_behavior != 'f->f':
        base_race = random.choice(minorities)
        base_num_ecs = random.choice(good_num_ecs)
        
        base_letters = random.choice(settings['num_letters'])
        base_gpa = random.choice(settings['gpa'])
        src_letters = random.choice(settings['num_letters'])
        src_gpa = random.choice(settings['gpa'])
        
        if ctf_behavior == 't->t':
            while not (base_letters >= 2 and base_gpa >= 3.0):
                base_letters = random.choice(settings['num_letters'])
                base_gpa = random.choice(settings['gpa'])
                
            while not (src_letters >= 2 and src_gpa >= 3.0):
                src_letters = random.choice(settings['num_letters'])
                src_gpa = random.choice(settings['gpa'])
                
            base_label = src_label = 'Yes'
                
        elif ctf_behavior == 't->f':
            while not (base_letters >= 2 and base_gpa >= 3.0):
                base_letters = random.choice(settings['num_letters'])
                base_gpa = random.choice(settings['gpa'])
            
            while src_letters >= 2 and src_gpa >= 3.0:
                src_letters = random.choice(settings['num_letters'])
                src_gpa = random.choice(settings['gpa'])
                
            base_label = 'Yes'
            src_label = 'No'
                
        elif ctf_behavior == 'f->t':
            while base_letters >= 2 and base_gpa >= 3.0:
                base_letters = random.choice(settings['num_letters'])
                base_gpa = random.choice(settings['gpa'])
                
            while not (src_letters >= 2 and src_gpa >= 3.0):
                src_letters = random.choice(settings['num_letters'])
                src_gpa = random.choice(settings['gpa'])
                
            base_label = 'No'
            src_label = 'Yes'
                
    else:
        base_race = random.choice(settings['race'])
        if base_race == 'White':
            base_num_ecs = random.choice(settings['num_ecs'])
        else:
            base_num_ecs = 0
        
        base_letters = random.choice(settings['num_letters'])
        base_gpa = random.choice(settings['gpa'])
        src_letters = random.choice(settings['num_letters'])
        src_gpa = random.choice(settings['gpa'])
        
        base_label = src_label = 'No'
        
    base_settings['race'] = base_race
    base_settings['num_ecs'] = base_num_ecs
    
    src_settings = base_settings.copy()
    src_settings['num_letters'] = src_letters
    src_settings['gpa'] = src_gpa
    
    base_settings['num_letters'] = base_letters
    base_settings['gpa'] = base_gpa
    
    return base_settings, src_settings, base_label, src_label

def format_prompt(template, candidate, 
                  dataset: Union['full', 'short'] = 'full'):
    if dataset == 'full':
        prompt = template.format(
            pronoun_pos = candidate['pronoun_pos'],
            pronoun = candidate['pronoun'],
            gender = candidate['gender'],
            race = candidate['race'],
            income = candidate['income'],
            geography = candidate['geography'],
            school = candidate['school'],
            gpa = candidate['gpa'],
            sat = candidate['sat'],
            num_ecs = candidate['num_ecs'],
            num_pres = candidate['num_pres'],
            letters_quality = candidate['letters_quality'],
            topic = candidate['topic']
        )
    else:
        prompt = template.format(
            race = candidate['race'],
            gpa = candidate['gpa'],
            num_ecs = candidate['num_ecs'],
            num_letters = candidate['num_letters']
        )
    return prompt

"""
Tokenize the label. The returned token index is somewhat
arbitrary because it depends on the tokenizer. In this case
it is the LLaMA tokenizer.
"""
def format_label(label_eng):
    if label_eng == 'Yes':
        return 8241
    else:
        return 3782

"""SCRIPT STARTS HERE"""

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_size", help="Specify the size of the dataset.")
parser.add_argument("--dataset_type", help="""Specify which causal variable 
                                            you are aligning with. Options:
                                            - race_variable
                                            - p_variable""")
parser.add_argument("--save_path", help="""Path to save the resulting dataset. 
                                        Should end in a directory.""")

args = parser.parse_args()
ds_type = args.dataset_type
ds_size = int(args.dataset_size)
save_path = args.save_path

template = open('./prompts/ug_admissions_short.txt').read()

if ds_type == "race_variable":
    sample_ctf_func = sample_one_ctf
elif ds_type == "p_variable":
    sample_ctf_func = sample_one_ctf_p

ctf_examples = [sample_ctf_func(BIOS_SETTINGS_SHORT) for _ in range(ds_size)]

dataset_dict = {
    'base': [format_prompt(template, ex[0], dataset='short') for ex in ctf_examples],
    'source': [format_prompt(template, ex[1], dataset='short') for ex in ctf_examples],
    'base_label': [format_label(ex[2]) for ex in ctf_examples],
    'src_label': [format_label(ex[3]) for ex in ctf_examples]
}

dataset_all = Dataset.from_dict(dataset_dict)
ds_train_test = dataset_all.train_test_split(test_size=0.2)
ds_test = ds_train_test['test']
ds_train_dev = ds_train_test['train'].train_test_split(test_size=0.2)
ds_train = ds_train_dev['train']
ds_dev = ds_train_dev['test']

df_train = pd.DataFrame.from_dict(ds_train)
df_dev = pd.DataFrame.from_dict(ds_dev)
df_test = pd.DataFrame.from_dict(ds_test)

os.makedirs(save_path, exist_ok=True)
df_train.to_csv(os.path.join(save_path, "train.csv"), index=False)
df_dev.to_csv(os.path.join(save_path, "dev.csv"), index=False)
df_test.to_csv(os.path.join(save_path, "test.csv"), index=False)