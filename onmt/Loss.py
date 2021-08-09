#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

import onmt
import onmt.io
# import opts

TGT_VOCAB_SIZE = 606


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None, \
                p_prob=None, p_target=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns, stage1=True):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          stage1: is it stage1
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt2.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, s_prob=None, s_target=None,
                             retain_graph=False):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        # shard_state = self._make_shard_state(batch, output, range_, attns)
        if s_prob is None and s_target is None:
            shard_state = self._make_shard_state(batch, output, range_, attns)
        else:
            shard_state = self._make_shard_state(
                batch, output, range_, attns, s_prob, s_target)
        shards_list = shards(
            shard_state,
            shard_size,
            retain_graph=retain_graph)
        for shard in shards_list:
            loss, stats = self._compute_loss(batch, **shard)
            if s_prob is not None:
                selection_loss = self._compute_selection_loss(
                    shard["p_output"], shard["p_target"])
                loss = loss + selection_loss * 0.05
            loss.div(normalization).backward()
            batch_stats.update(stats)

        return batch_stats

    def _compute_selection_loss(self, output, target):
        """
        compute dynamic selection loss

        :param output:
        :param target:
        :return:
        """
        neg_tgt_idx = target.eq(0).float()
        neg_prob = neg_tgt_idx.mul(output)
        neg_prob = neg_tgt_idx.mul(1 - neg_prob)

        pos_tgt_idx = target.eq(1).float()
        pos_prob = pos_tgt_idx.mul(output)

        out = pos_prob + neg_prob

        # masked padding
        out += out.eq(0).float()

        selection_loss = -out.log().div(out.size(2)).sum()

        return selection_loss

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()

        return onmt.Statistics(loss.item(), non_padding.sum(), num_correct)

    def _bottle(self, v):
        """

        :param v:
        :return:
        """
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        """

        :param v:
        :param batch_size:
        :return:
        """
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0, decoder_type='rnn'):
        """

        :param generator:
        :param tgt_vocab:
        :param normalization:
        :param label_smoothing:
        :param decoder_type:
        """
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)
        self.decoder_type = decoder_type
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.

            if self.decoder_type == 'pointer':
                tgt_vocab_len = TGT_VOCAB_SIZE
            else:
                tgt_vocab_len = len(tgt_vocab)

            self.criterion = nn.KLDivLoss(size_average=None)
            # one_hot = torch.randn(1, len(tgt_vocab))
            # one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot = torch.randn(1, tgt_vocab_len)
            one_hot.fill_(label_smoothing / (tgt_vocab_len - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            if self.decoder_type == 'pointer':
                weight = torch.ones(TGT_VOCAB_SIZE)
            else:
                weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=None)
        self.confidence = 1.0 - label_smoothing

    def _make_shard_state(self, batch, output, range_, attns=None, \
                p_prob=None, p_target=None):
        """

        :param batch:
        :param output:
        :param range_:
        :param attns:
        :return:
        """
        if self.decoder_type == 'pointer':
            return {
                "output": attns['std'],
                "target": batch.tgt1_planning[range_[0] + 1: range_[1]]
            }
        else:
            '''
            assert False
            return {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1]],
            }
            '''
            return {
                "output": output,
                "target": batch.tgt2[range_[0] + 1: range_[1]],
            }

    def _compute_loss(self, batch, output, target):
        """

        :param batch:
        :param output:
        :param target:
        :return:
        """
        if self.decoder_type == 'pointer':
            scores = self._bottle(output)
        else:
            scores = self.generator(self._bottle(output))
        gtruth = target.contiguous().view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)
        loss = self.criterion(scores, gtruth)
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        if self.decoder_type != "pointer":

            pred_score = torch.cat([p_score[gt]
                                    for gt, p_score in zip(gtruth, scores)])
            log_regular_value = self._loss_regular(
                pred_score, gtruth, batch.batch_size, self.padding_idx)

            # loss.data = loss.data + log_regular_value.data
            loss_alpha = 0.5
            # loss_alpha = 0.1
            loss = loss * loss_alpha + log_regular_value
            # loss = loss * loss_alpha

        stats = self._stats(loss_data, scores.data, target.contiguous().view(-1).data)
        return loss, stats

    def _loss_regular(self, pred, target, batch_size, padding_idx):
        """
        compute loss regular item

        :param pred:
        :param target:
        :param batch_size:
        :param padding_idx:
        :return:
        """
        pad_num = 0
        mode = pred.size()[0] / batch_size % 4
        if mode != 0:
            pad_num = int(4 - mode)
            pad_data = pred[0:batch_size] * 0 + padding_idx
            pad_data = torch.cat([pad_data for _ in range(pad_num)])
            pred = torch.cat([pred, pad_data])
            target = torch.cat([target, pad_data.long()])

        pred = pred.view(-1, 4, batch_size)
        target = target.view(-1, 4, batch_size)
        pred = pred.mul(target.ne(padding_idx).float())

        lens_pred = target.ne(padding_idx).float().sum(1, True)
        lens_pred = lens_pred + lens_pred.eq(0).float()

        sum_pred = pred.sum(1, True)

        avg_pred = torch.div(sum_pred, lens_pred)

        alpha = 0.05

        regular_value = torch.log(torch.abs(pred - avg_pred).sum() + 1) * alpha

        return regular_value


def filter_shard_state(state, requires_grad=True, volatile=False):
    """

    :param state:
    :param requires_grad:
    :param volatile:
    :return:
    """
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=requires_grad,
                             volatile=volatile)
            yield k, v


def shards(state, shard_size, eval=False, retain_graph=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    # print "shards begin"
    if eval:
        yield filter_shard_state(state, False, True)
    else:

        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))
        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        # keys, values = zip(*((k, torch.split(v, shard_size))
        #                     for k, v in non_none.items()))    修改

        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))
        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.

        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        # variables contain output / copy_attn
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads, retain_graph=retain_graph)
