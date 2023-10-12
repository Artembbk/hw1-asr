import logging
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {
        'spectrogram': [],
        'text_encoded': [],
        'text': [],
        'text_encoded_length': []
    }

    for item in dataset_items: 
        result_batch['spectrogram'].append(item['spectrogram'].squeeze(0).T)
        result_batch['text_encoded'].append(item['text_encoded'].view(-1,))
        result_batch['text'].append(item['text'])
        result_batch['text_encoded_length'].append(item['text_encoded'].shape[0])
    
    result_batch['text_encoded'] = pad_sequence(result_batch['text_encoded'], True, 0)
    result_batch['spectrogram'] = pad_sequence(result_batch['spectrogram'], True, 0).transpose(1, 2)
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'])
    
    return result_batch