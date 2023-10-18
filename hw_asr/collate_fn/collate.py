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
        'text_encoded_length': [],
        'spectrogram_length': [],
        'audio_path': []
    }

    for item in dataset_items: 
        result_batch['spectrogram'].append(item['spectrogram'].squeeze(0).T)
        result_batch['text_encoded'].append(item['text_encoded'].view(-1,))
        result_batch['text'].append(item['text'])
        result_batch['audio_path'].append(item['audio_path'])
        result_batch['text_encoded_length'].append(item['text_encoded'].shape[1])
        result_batch['spectrogram_length'].append(result_batch['spectrogram'][-1].shape[0])

    
    result_batch['text_encoded'] = pad_sequence(result_batch['text_encoded'], True, 0)
    result_batch['spectrogram'] = pad_sequence(result_batch['spectrogram'], True, 0).transpose(1, 2)
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'])
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'])
    
    return result_batch