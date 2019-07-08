#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:04:14 2019

@author: xiongyi
"""
import logging
#logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
#logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
#logging.info('test')
import os, sys
import torch
## Set PATHs
#PATH_TO_SENTEVAL = './senteval'
#PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL,'../data')
#
#
## PATH_TO_VEC = 'glove/glove.840B.300d.txt'
#
## import SentEval
#sys.path.insert(0, PATH_TO_SENTEVAL)
#import senteval
#
## Set up logger
#params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
#params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
#                             'tenacity': 3, 'epoch_size': 2}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, unique_id, text_a, text_b=None):
        """Constructs a InputExample.

        Args:
            unique_id: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeaturesNL(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        
def convert_examples_to_features_NL(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #print ('tokens', tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeaturesNL(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features
       

# SentEval prepare and batcher
def prepare(params, samples):
    params.batch_size = 64
    #print ('samples.shape', samples.shape)
    return

def batcher(params, batch):
    #print ('batch size' ,len(batch))
    
    #assert len(batch[0]) == 2, 'batch format error'
    batch = [ ' '.join(dat).lower()   if dat != [] else '.' for dat in batch  ]
    #print ('batch size' ,len(batch))
    #print ('batch[0]' ,batch[0])
    #print ('batch', batch)
    examples = []
    unique_id = 0
    #print ('batch size ', len(batch))
    for dat in batch:
        sent = dat[0].strip()
        text_b = None
        text_a = sent
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        unique_id += 1  
    
    features = convert_examples_to_features_NL(examples, params['bert'].seq_length, params['bert'].tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(params['bert'].device)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(params['bert'].device)
    #print ('all_input_ids.shape', all_input_ids.shape)
    ###get z_vec
    #get the output of previous layer
    
    embeddings,pooled_output = params['bert'].bert(all_input_ids, token_type_ids=None, attention_mask=all_input_mask, output_all_encoded_layers=False)
    #print ('embeddings.shape', embeddings.shape)
    embeddings = embeddings[:,0,:].detach().cpu().numpy()
    return embeddings


def probe(model, tokenizer, device, max_seq_length, batcher, prepare, PATH_TO_SENTEVAL, PATH_TO_DATA, transfer_tasks = ['WordContent']):
    sys.path.insert(0, PATH_TO_SENTEVAL)
    import senteval
    
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 3}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                 'tenacity': 3, 'epoch_size': 2}
    params_senteval['bert'] = model
    params_senteval['bert'].seq_length = max_seq_length    
    params_senteval['bert'].tokenizer = tokenizer
    params_senteval['bert'].device = device
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    results = se.eval(transfer_tasks)
    print(results)
    return results
    
    
    
    
    
    
    
    
    
