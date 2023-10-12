from typing import List, NamedTuple

import torch


from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        inds_tensor = torch.tensor(inds)
        mask = torch.cat([torch.tensor([True]), inds_tensor[:-1] != inds_tensor[1:]])

        result_tensor = inds_tensor[mask]
        return self.decode(list(result_tensor[result_tensor != self.char2ind[self.EMPTY_TOK]]))
        
    def extend_and_merge(self, frame, hypos):
        new_hypos = {}
        for next_ind, next_char_prob in enumerate(frame):
            next_char = self.ind2char[next_ind]
            for hypo in hypos:
                if next_char == self.EMPTY_TOK or next_char == hypo.text[-1]:
                    new_hypos[hypo.text] += hypo.prob * next_char_prob

        return [Hypothesis(new_hypo.text, new_hypo.prob) for new_hypo in new_hypos]
        

    def truncate(self, new_hypos, beam_size):
        sorted_hypos = new_hypos.sort(key=lambda x: -x.prob)
        return sorted_hypos[:beam_size]
                



    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        hypos.append(Hypothesis("", 1.0))
        for i, frame, frame_length in enumerate(zip(probs, probs_length)):
            if i == probs_length:
                break
            new_hypos = self.extend_and_merge(frame, hypos)
            hypos = self.truncate(new_hypos, beam_size)
        return hypos
