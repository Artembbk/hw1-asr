from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer
import time


class BeamCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        start_time = time.time()
        for log_probs_1, log_probs_length_1, target_text in zip(log_probs.cpu(), lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_beam_search(torch.exp(log_probs_1), log_probs_length_1)[0].text
            cers.append(calc_cer(target_text, pred_text))
        end_time = time.time()
        print("cer_time", end_time - start_time)

        return sum(cers) / len(cers)
