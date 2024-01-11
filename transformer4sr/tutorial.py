#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Transformer 4 Symbolic Regression

# In[1]:


import torch
import sympy
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from model.transformer_model import TransformerModel
from model._utils import count_nb_params
from model._utils import is_tree_complete
from model._utils import translate_integers_into_tokens
from datasets._utils import from_sequence_to_sympy


# ### 1. Instantiate the Transformer Model
# 
# We begin by instantiating an empty Transformer.  
# Count and print its number of total trainable parameters (for information).

# In[2]:


# First reload big model
transformer = TransformerModel(
    enc_type='mix',
    nb_samples=50,  # Number of samples par dataset
    max_nb_var=7,  # Max number of variables
    d_model=256,
    vocab_size=18+2,  # len(vocab) + padding token + <SOS> token
    seq_length=30,  # vocab_size + 1 - 1 (add <SOS> but shifted right)
    h=4,
    N_enc=4,
    N_dec=8,
    dropout=0.25,
)
total_nb_params = count_nb_params(transformer, print_all=False)
print(f'Total number params = {total_nb_params}')


# ### 2. Load pre-trained state, using `OrderedDict` to match module names exactly
# 
# Specify the path with the best Transformer model's weights.  
# Load the weights, and remove `module.` from their name to match the new names.  
# Substitute the weights from the empty Transformer model to the best weights.  

# In[3]:


PATH_WEIGHTS = 'best_model_weights/mix_label_smoothing/model_weights.pt'
hixon_state_dict = torch.load(PATH_WEIGHTS, map_location=torch.device('cpu'))

my_state_dict = OrderedDict()
for key in hixon_state_dict.keys():
    assert key[:7]=="module."
    my_state_dict[key[7:]] = hixon_state_dict[key]

out = transformer.load_state_dict(my_state_dict, strict=True)
transformer.eval()  # deactivate training mode (important)
print(out)  # This should print <All keys matched susccessfully>


# ### 3. A function to decode in an auto-regressive fashion
# 
# This function will be used to decode with the Transformer model in an auto-regression fashion.  
# Start by feeding the numerical tabular dataset to the Encoder. The `encoder_output` is the same for the whole procedure.  
# Initiate the Decoder with the start of sequence `<SOS>` token in first position.  
# Then loop: countinue decoding until the equation tree is complete.

# In[4]:


def decode_with_transformer(transformer, dataset):
    """
    Greedy decode with the Transformer model.
    Decode until the equation tree is completed.
    Parameters:
      - transformer: torch Module object
      - dataset: tabular dataset
      shape = (batch_size=1, nb_samples=50, nb_max_var=7, 1)
    """
    encoder_output = transformer.encoder(dataset)  # Encoder output is fixed for the batch
    seq_length = transformer.decoder.positional_encoding.seq_length
    decoder_output = torch.zeros((dataset.shape[0], seq_length+1), dtype=torch.int64)  # initialize Decoder output
    decoder_output[:, 0] = 1
    is_complete = torch.zeros(dataset.shape[0], dtype=torch.bool)  # check when decoding is finished
    
    for n1 in range(seq_length):
        padding_mask = torch.eq(decoder_output[:, :-1], 0).unsqueeze(1).unsqueeze(1)
        future_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        mask_dec = torch.logical_or(padding_mask, future_mask)
        temp = transformer.decoder(
            target_seq=decoder_output[:, :-1],
            mask_dec=mask_dec,
            output_enc=encoder_output,
        )
        temp = transformer.last_layer(temp)
        decoder_output[:, n1+1] = torch.where(is_complete, 0, torch.argmax(temp[:, n1], axis=-1))
        for n2 in range(dataset.shape[0]):
            if is_tree_complete(decoder_output[n2, 1:]):
                is_complete[n2] = True
    return decoder_output


# ### 4. Generate your own tabular dataset
# 
# Instantiate the necessary SymPy symbols.  
# Design custom ground-truth equation, and print the ground-truth LATEX formula.  
# Sample numerical values, and create a random tabular `dataset` following the ground-truth equation.  
# Generate `encoder_input` and feed it to the Transformer model using the above decoding function.  
# Decode the predict sequence of tokens into SymPy equation, and print it!

# In[16]:


# Instantiate the SymPy symbols
C, y, x1, x2, x3 = sympy.symbols('C, y, x1, x2, x3', real=True, positive=True)

# Create your own ground truth
y = 25 * x1 + x2 * sympy.log(x1)
print('The ground truth is:')
y


# In[17]:


# Sample numerical values for x1 and x2 (add more columns if necessary, otherwise zeros)
x1_values = np.power(10.0, np.random.uniform(-1.0, 1.0, size=50))
x2_values = np.power(10.0, np.random.uniform(-1.0, 1.0, size=50))

# Evaluate the ground_truth
f = sympy.lambdify([x1, x2], y)
y_values = f(x1_values, x2_values)


# In[18]:


# Make tabular dataset
dataset = np.zeros((50, 7))
dataset[:, 0] = y_values
dataset[:, 1] = x1_values
dataset[:, 2] = x2_values

# Generate input for the Encoder using torch.Tensor object
encoder_input = torch.Tensor(dataset).unsqueeze(0).unsqueeze(-1)


# In[19]:


decoder_output = decode_with_transformer(transformer, encoder_input)
decoder_tokens = translate_integers_into_tokens(decoder_output[0])
sympy_pred = from_sequence_to_sympy(decoder_tokens)
print(sympy_pred)


