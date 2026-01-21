"""Utility functions for student-implemented loss computations.

The training entry point expects a callable named `compute_loss_from_logits`.
Students should implement the function so that it takes model logits and
ground truth labels and returns a scalar loss tensor.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.trainer_pt_utils import LabelSmoother

from transformers.modeling_outputs import CausalLMOutputWithPast


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def compute_loss_from_logits(
    outputs: CausalLMOutputWithPast,
    labels: Optional[torch.Tensor],
    num_items_in_batch: int,
) -> torch.Tensor:
    """Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        ignore_index: Label id that should be ignored when computing the loss. The
            trainer passes HuggingFace's default ignore index (-100).

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.

    Students should implement this function by computing the cross-entropy loss
    from the raw logits. You may not call `torch.nn.CrossEntropyLoss`; instead,
    derive the loss explicitly using a log-softmax over the vocabulary dimension.
    """

    # raise NotImplementedError("Implement token-level cross-entropy using the logits.")
    logits = outputs.logits
    return cross_entropy_loss(logits, labels, num_items_in_batch=num_items_in_batch)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int,
) -> torch.Tensor:
    """
    Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        num_items_in_batch: Number of valid items in batch for normalization.

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.
    """
    
    assert logits.ndim == 3
    assert labels.ndim == 2
    assert logits.shape[:2] == labels.shape
    
    B, seq_len, vocab_size = logits.shape
    device = logits.device
    dtype = logits.dtype
    
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    valid_mask = shift_labels.ne(IGNORE_TOKEN_ID)
    
    if not torch.any(valid_mask):
        # 否则会除 0
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    
    max_logits = shift_logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(shift_logits - max_logits).sum(dim=-1, keepdim=True)) + max_logits
    log_probs = shift_logits - log_sum_exp
    
    # shift_labels 里有 -100，直接拿去 gather 会报错, 把被 mask 的位置临时改成 0（反正后面会乘 mask 清零）
    safe_labels = torch.where(valid_mask, shift_labels, torch.zeros_like(shift_labels))
    gold_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    
    nll = -gold_log_probs
    # 不参与的位置清零
    nll = nll * valid_mask.to(dtype)

    loss_sum = nll.sum()
    
    denom = max(int(num_items_in_batch), 1)
    loss = loss_sum / denom

    return loss



