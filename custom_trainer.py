from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt

import onmt.modules


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Variable): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src1', 'src2', 'tgt1', 'tgt2', 'tgt2_ptrs']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    if data_type == 'text':
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        """

        :param loss:
        :param n_words:
        :param n_correct:
        """
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        """

        :param stat:
        :return:
        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        """

        :return:
        """
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        """

        :return:
        """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """

        :return:
        """
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batches (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch, n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        """

        :param prefix:
        :param experiment:
        :param lr:
        :return:
        """
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper", self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, epoch):
        """

        :param prefix:
        :param writer:
        :param lr:
        :param epoch:
        :return:
        """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/ppl", self.ppl(), epoch)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), epoch)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, epoch)
        writer.add_scalar(prefix + "/lr", lr, epoch)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, model2, train_loss, valid_loss,
                 train_loss2, valid_loss2, train_loss3,
                 valid_loss3, optim, optim2, trunc_size=0,
                 shard_size=32, data_type='text', norm_method="sents",
                 grad_accum_count=1, cuda=False, finetune=False):
        # Basic attributes.
        self.model = model
        self.model2 = model2
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.train_loss2 = train_loss2
        self.valid_loss2 = valid_loss2

        self.train_loss3 = train_loss3
        self.valid_loss3 = valid_loss3

        self.optim = optim
        self.optim2 = optim2
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.cuda = cuda
        self.finetune = finetune

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()
        self.model2.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        total_stats2 = Statistics()
        report_stats2 = Statistics()

        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset
            self.train_loss2.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, total_stats2, report_stats2, normalization)

                if report_func is not None:
                    report_stats = report_func(
                        epoch, idx, num_batches,
                        total_stats.start_time, self.optim.lr,
                        report_stats)
                    report_stats2 = report_func(
                        epoch, idx, num_batches,
                        total_stats2.start_time, self.optim2.lr,
                        report_stats2)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

                # break

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                true_batchs, total_stats,
                report_stats, total_stats2, report_stats2, normalization)
            true_batchs = []

        return total_stats, total_stats2

    def train_ft(self, train_iter, epoch, report_func=None):
        """ Train next epoch in finetuned way.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        total_stats2 = Statistics()
        report_stats2 = Statistics()
        total_stats3 = Statistics()
        report_stats3 = Statistics()

        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset
            self.train_loss2.cur_dataset = cur_dataset
            self.train_loss3.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation_v2(
                    true_batchs, total_stats,
                    report_stats, total_stats2, report_stats2, normalization,
                    total_stats3, report_stats3)

                if report_func is not None:
                    report_stats = report_func(
                        epoch, idx, num_batches,
                        total_stats.start_time, self.optim.lr,
                        report_stats)
                    report_stats2 = report_func(
                        epoch, idx, num_batches,
                        total_stats2.start_time, self.optim2.lr,
                        report_stats2)
                    report_stats3 = report_func(
                        epoch, idx, num_batches,
                        total_stats3.start_time, self.optim2.lr,
                        report_stats3)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation_v2(
                true_batchs, total_stats,
                report_stats, total_stats2, report_stats2, normalization,
                total_stats3, report_stats3)
            true_batchs = []

        return total_stats, total_stats2, total_stats3

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        self.model2.eval()

        stats = Statistics()
        stats2 = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset
            self.valid_loss2.cur_dataset = cur_dataset
            batch.tgt2 = batch.tgt2[0]

            src = make_features(batch, 'src1', self.data_type)
            self.tt = torch.cuda if self.cuda else torch
            src_lengths = self.tt.LongTensor(
                batch.src1.size()[1]).fill_(
                batch.src1.size()[0])

            tgt = batch.tgt1_planning.unsqueeze(2)
            # F-prop through the model.
            outputs, attns, _, memory_bank = self.model(src, tgt, src_lengths)
            # Compute loss.
            print("valid stage1 loss1 begin")
            batch_stats = self.valid_loss.monolithic_compute_loss(
                batch, outputs, attns, stage1=True)
            # Update statistics.
            stats.update(batch_stats)
            print("valid stage1 loss1 end")

            inp_stage2 = tgt[1:-1]
            index_select = [
                torch.index_select(
                    a, 0, i).unsqueeze(0) for a, i in zip(
                    torch.transpose(
                        memory_bank, 0, 1), torch.t(
                        torch.squeeze(
                            inp_stage2, 2)))]
            emb = torch.transpose(torch.cat(index_select), 0, 1)
            _, src_lengths = batch.src2
            tgt = make_features(batch, 'tgt2')
            # F-prop through the model.
            outputs, attns, _, _ = self.model2(emb, tgt, src_lengths)
            # Compute loss.
            print("valid stage2 loss begin")
            batch_stats = self.valid_loss2.monolithic_compute_loss(
                batch, outputs, attns, stage1=False)
            # Update statistics.
            stats2.update(batch_stats)
            print("valid stage2 loss end")

        # Set model back to training mode.
        self.model.train()
        self.model2.train()

        return stats, stats2

    def validate_v2(self, valid_iter):
        """ Validate model for dynamic planning.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        self.model2.eval()

        stats = Statistics()
        stats2 = Statistics()
        stats3 = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset
            self.valid_loss2.cur_dataset = cur_dataset
            self.valid_loss3.cur_dataset = cur_dataset

            _, tgt_lengths = batch.tgt2
            tgt_lengths = tgt_lengths - 1
            batch.tgt2 = batch.tgt2[0]
            src2_all, src2_all_lengths = batch.src2_all
            src2_all = src2_all.unsqueeze(2)
            src2_all_lengths = src2_all_lengths - 1
            src = make_features(batch, 'src1', self.data_type)
            self.tt = torch.cuda if self.cuda else torch
            src_lengths = self.tt.LongTensor(
                batch.src1.size()[1]).fill_(
                batch.src1.size()[0])

            tgt = batch.tgt1_planning.unsqueeze(2)
            # F-prop through the model.
            outputs, attns, _, memory_bank = self.model(src, tgt, src_lengths)
            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                batch, outputs, attns, stage1=True)
            # Update statistics.
            stats.update(batch_stats)

            inp_stage2 = tgt[1:-1]
            index_select = [
                torch.index_select(
                    a, 0, i).unsqueeze(0) for a, i in zip(
                    torch.transpose(
                        memory_bank, 0, 1), torch.t(
                        torch.squeeze(
                            inp_stage2, 2)))]
            emb = torch.transpose(torch.cat(index_select), 0, 1)
            _, src_lengths = batch.src2

            tgt = make_features(batch, 'tgt2')

            # F-prop through the model.

            dec_state = None
            decoder1_res, decoder2_res = self.model2.forward_V2(
                emb, tgt, src2_all, src_lengths, tgt_lengths, dec_state)
            outputs, attns, dec_state, _ = decoder1_res
            outputs2, attns2, dec_state2, _ = decoder2_res

            batch_stats = self.valid_loss2.monolithic_compute_loss(
                batch, outputs, attns, stage1=False)
            # Update statistics.
            stats2.update(batch_stats)

            # modification for new decoder outputs' loss
            batch.tgt2 = batch.src2_all[0]

            batch_stats = self.valid_loss3.monolithic_compute_loss(
                batch, outputs2, attns2, stage1=False)
            stats3.update(batch_stats)

        # Set model back to training mode.
        self.model.train()
        self.model2.train()

        return stats, stats2, stats3

    def epoch_step(self, ppl, ppl2, epoch):
        """

        :param ppl:
        :param ppl2:
        :param epoch:
        :return:
        """
        self.optim.update_learning_rate(ppl, epoch)
        self.optim2.update_learning_rate(ppl2, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats, valid_stats2):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_stage1_acc_%.4f_ppl_%.4f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

        real_model = (self.model2.module
                      if isinstance(self.model2, nn.DataParallel)
                      else self.model2)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim2,
        }
        torch.save(checkpoint,
                   '%s_stage2_acc_%.4f_ppl_%.4f_e%d.pt'
                   % (opt.save_model, valid_stats2.accuracy(),
                      valid_stats2.ppl(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats, report_stats,
                               total_stats2, report_stats2, normalization):
        """
        calculdate gradient and accumulation

        :param true_batchs:
        :param total_stats:
        :param report_stats:
        :param total_stats2:
        :param report_stats2:
        :param normalization:
        :return:
        """
        if self.grad_accum_count > 1:
            assert False
            self.model.zero_grad()
        for batch in true_batchs:
            # Stage 1
            target_size = batch.tgt1.size(0)
            trunc_size = target_size
            dec_state = None
            src = make_features(batch, 'src1', self.data_type)
            self.tt = torch.cuda if self.cuda else torch
            src_lengths = self.tt.LongTensor(
                batch.src1.size()[1]).fill_(
                batch.src1.size()[0])

            for j in range(0, target_size - 1, trunc_size):
                # setting to value of tgt_planning
                tgt = batch.tgt1_planning[j: j + trunc_size].unsqueeze(2)
                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state, memory_bank = \
                    self.model(src, tgt, src_lengths, dec_state)
                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization, retain_graph=True)
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            # Stage 2
            target_size = batch.tgt2[0].size(0)
            src_size = batch.src2[0].size(0) + 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                assert False
                trunc_size = target_size
            trunc_size = target_size
            dec_state = None
            _, src_lengths = batch.src2
            _, tgt_lengths = batch.tgt2
            batch.tgt2 = batch.tgt2[0]
            tgt_lengths = tgt_lengths - 1
            report_stats2.n_src_words += src_lengths.sum()

            # memory bank is of size src_len*batch_size*dim, inp_stage2 is of
            # size inp_len*batch_size*1
            inp_stage2 = tgt[1:-1]
            index_select = [
                torch.index_select(
                    a, 0, i).unsqueeze(0) for a, i in zip(
                    torch.transpose(
                        memory_bank, 0, 1), torch.t(
                        torch.squeeze(
                            inp_stage2, 2)))]

            # use stage1 encoder hidden states as emb, size: (length,
            # batch_size, hidden_size)
            emb = torch.transpose(torch.cat(index_select), 0, 1)
            if self.data_type == 'text':
                tgt_outer = make_features(batch, 'tgt2')
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model2.zero_grad()
                outputs, attns, dec_state, _ = \
                    self.model2(emb, tgt, src_lengths, dec_state)

                plan_prob, plan_target = self._plan_score_target(batch)
                # retain_graph is false for the final truncation
                retain_graph = (j + trunc_size) < (target_size - 1)
                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss2.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization,
                    s_prob=plan_prob, s_target=plan_target,
                    retain_graph=retain_graph)
                total_stats2.update(batch_stats)
                report_stats2.update(batch_stats)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim2.step()

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()
            # 4. Update the parameters and statistics.
            self.optim.step()
        if self.grad_accum_count > 1:
            assert False
            self.optim.step()

    def _gradient_accumulation_v2(self, true_batchs, total_stats,
                                  report_stats, total_stats2, report_stats2,
                                  normalization, total_stats3, report_stats3):
        """
        calculate gradient and accumulation for dynamic planning

        :param true_batchs:
        :param total_stats:
        :param report_stats:
        :param total_stats2:
        :param report_stats2:
        :param normalization:
        :param total_stats3:
        :param report_stats3:
        :return:
        """

        if self.grad_accum_count > 1:
            assert False
            self.model.zero_grad()
        for batch in true_batchs:
            # Stage 1
            target_size = batch.tgt1.size(0)
            trunc_size = target_size
            dec_state = None
            src = make_features(batch, 'src1', self.data_type)
            self.tt = torch.cuda if self.cuda else torch
            src_lengths = self.tt.LongTensor(
                batch.src1.size()[1]).fill_(
                batch.src1.size()[0])
            for j in range(0, target_size - 1, trunc_size):
                # setting to value of tgt_planning
                tgt = batch.tgt1_planning[j: j + trunc_size].unsqueeze(2)
                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state, memory_bank = \
                    self.model(src, tgt, src_lengths, dec_state)
                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization, retain_graph=True)
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
                print(batch_stats.n_words, batch_stats.n_correct)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()
            # Stage 2
            # target_size = batch.tgt2.size(0)
            target_size = batch.tgt2[0].size(0)
            src_size = batch.src2[0].size(0) + 1
            src2_all_size = batch.src2_all[0].size(0) + 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                assert False
                trunc_size = target_size
            trunc_size = target_size
            dec_state = None
            _, src_lengths = batch.src2
            _, tgt_lengths = batch.tgt2
            batch.tgt2 = batch.tgt2[0]
            src2_all, src2_all_lengths = batch.src2_all
            src2_all = src2_all.unsqueeze(2)
            src2_all_lengths = src2_all_lengths - 1
            tgt_lengths = tgt_lengths - 1
            report_stats2.n_src_words += src_lengths.sum()

            # memory bank is of size src_len*batch_size*dim, inp_stage2 is of
            # size inp_len*batch_size*1
            inp_stage2 = tgt[1:-1]
            index_select = [
                torch.index_select(
                    a, 0, i).unsqueeze(0) for a, i in zip(
                    torch.transpose(
                        memory_bank, 0, 1), torch.t(
                        torch.squeeze(
                            inp_stage2, 2)))]

            # use stage1 encoder hidden states as emb, size: (length,
            # batch_size, hidden_size)
            emb = torch.transpose(torch.cat(index_select), 0, 1)

            if self.data_type == 'text':
                tgt_outer = make_features(batch, 'tgt2')

            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model2.zero_grad()

                decoder1_res, decoder2_res = self.model2.forward_V2(
                    emb, tgt, src2_all, src_lengths, tgt_lengths, dec_state)

                outputs, attns, dec_state, _ = decoder1_res
                outputs2, attns2, dec_state2, _ = decoder2_res
                plan_prob, plan_target = self._plan_score_target(batch)

                # retain_graph is false for the final truncation
                retain_graph = (j + trunc_size) < (target_size - 1)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss2.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization,
                    s_prob=plan_prob, s_target=plan_target,
                    retain_graph=True)
                # trunc_size, self.shard_size, normalization,
                # retain_graph=retain_graph)

                total_stats2.update(batch_stats)
                report_stats2.update(batch_stats)

                # modification for new decoder outputs' loss
                batch.tgt2 = batch.src2_all[0]

                batch_stats = self.train_loss3.sharded_compute_loss(
                    batch, outputs2, attns2, j,
                    src2_all_size, self.shard_size, normalization, retain_graph=retain_graph)
                # src_size, self.shard_size, normalization,
                # retain_graph=retain_graph)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim2.step()

                total_stats3.update(batch_stats)
                report_stats3.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            # 4. Update the parameters and statistics.
            self.optim.step()
            # break

        if self.grad_accum_count > 1:
            assert False
            self.optim.step()

    def _plan_score_target(self, batch):
        """
        get dynamic content planning score and target

        :param batch:
        :return:
        """
        batch.tgt2_ptrs = batch.tgt2_ptrs[0][1:]
        batch.tgt1 = batch.tgt1[1:-1]
        tgt1_vocab = self.train_loss.cur_dataset.fields["tgt1"].vocab
        target1 = Variable(torch.zeros(batch.tgt1.size()))

        row, col = target1.size()

        for r in range(row):
            for c in range(col):
                # word = tgt1_vocab.itos[int(batch_tgt1[i, j])]
                word = tgt1_vocab.itos[int(batch.tgt1[r, c])]
                if word == "</s>" or word == "<blank>":
                    target1[r, c] = -1
                else:
                    target1[r, c] = int(word)

        plan_target = Variable(torch.zeros(batch.tgt2_ptrs.size())).cuda()

        tgt2_ptrs_vocab = self.train_loss.cur_dataset.fields["tgt2_ptrs"].vocab
        row, col = plan_target.size()

        for r in range(row):
            for c in range(col):
                word = tgt2_ptrs_vocab.itos[int(batch.tgt2_ptrs[r, c])]
                if word == "</s>":
                    plan_target[r, c] = 0
                elif word == "<blank>":
                    plan_target[r, c] = -1
                else:
                    plan_target[r, c] = int(word)
        selection_prob = self.model2.decoder.p_attn_alpha  # tgt_len, batch_size, src_len
        selection_target = Variable(torch.zeros(selection_prob.size())).cuda()
        tgt_len, batch_size, src_len = selection_target.size()
        for i in range(tgt_len):
            for k in range(batch_size):
                plan_target_idx = int(plan_target[i][k])
                if plan_target_idx == -1:
                    selection_target.data[i][k][:] = -1
                elif plan_target_idx == 0:
                    pass
                else:
                    # selection_target.data[i][k][plan_target_idx] = 1
                    selection_target.data[i][k] = target1[:, k].eq(
                        plan_target_idx).float().data
        return selection_prob, selection_target
