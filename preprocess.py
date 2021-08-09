import codecs
import os
import glob
import sys
import gc
import torch
import argparse
from functools import partial
from collections import Counter, defaultdict


import opts
from processdata import inputter as inputter
from processdata.logger import logger

import onmt
import onmt.io


def check_path(path):
    """

    :param path:
    :return:
    """
    path = os.path.abspath(path)
    path_dirname = os.path.dirname(path)
    if not os.path.exists(path_dirname):
        os.makedirs(path_dirname)


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def build_save_vocab(train_dataset, fields, opt):
    """

    :param train_dataset:
    :param fields:
    :param opt:
    :return:
    """
    fields = onmt.io.build_vocab(train_dataset, fields, opt.data_type,
                                 opt.share_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab_size,
                                     opt.tgt_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(onmt.io.save_fields_to_vocab(fields), vocab_file)


def build_save_text_dataset_in_shards(
        src_corpus,
        tgt_corpus,
        src_corpus2,
        tgt_corpus2,
        tgt2_ptrs_corpus,
        fields,
        corpus_type,
        opt,
        pointers):
    """
    Divide the big corpus into shards, and build dataset separately.
    This is currently only for data_type=='text'.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.

    To tackle this, we only read in part of the corpus file of size
    `max_shard_size`(actually it is multiples of 64 bytes that equals
    or is slightly larger than this size), and process it into dataset,
    then write it to disk along the way. By doing this, we only focus on
    part of the corpus at any moment, thus effectively reducing memory use.
    According to test, this method can reduce memory footprint by ~50%.

    Note! As we process along the shards, previous shards might still
    stay in memory, but since we are done with them, and no more
    reference to them, if there is memory tight situation, the OS could
    easily reclaim these memory.

    If `max_shard_size` is 0 or is larger than the corpus size, it is
    effectively preprocessed into one dataset, i.e. no sharding.

    NOTE! `max_shard_size` is measuring the input corpus size, not the
    output pt file size. So a shard pt file consists of examples of size
    2 * `max_shard_size`(source + target).

    :param src_corpus:
    :param tgt_corpus:
    :param src_corpus2:
    :param tgt_corpus2:
    :param tgt2_ptrs_corpus:
    :param fields:
    :param corpus_type:
    :param opt:
    :param pointers:
    :return:
    """

    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * (1024**2) and opt.max_shard_size == 0:
        print("Warning. The corpus %s is larger than 10M bytes, you can "
              "set '-max_shard_size' to process it by small shards "
              "to use less memory." % src_corpus)

    if opt.max_shard_size != 0:
        print(' * divide corpus into shards and build dataset separately'
              '(shard_size = %d bytes).' % opt.max_shard_size)


    ret_list = []
    src_shard = onmt.io.ShardedTextCorpusIterator(
        src_corpus, opt.src_seq_length_trunc,
        "src1", opt.max_shard_size)
    tgt_shard = onmt.io.ShardedTextCorpusIterator(
        tgt_corpus, opt.tgt_seq_length_trunc,
        "tgt1", opt.max_shard_size,
        assoc_iter=src_shard)
    src_shard2 = onmt.io.ShardedTextCorpusIterator(
        src_corpus2, opt.src_seq_length_trunc,
        "src2", opt.max_shard_size)
    tgt_shard2 = onmt.io.ShardedTextCorpusIterator(
        tgt_corpus2, opt.tgt_seq_length_trunc,
        "tgt2", opt.max_shard_size,
        assoc_iter=src_shard2)
    tgt2_ptrs_shard = onmt.io.ShardedTextCorpusIterator(
        tgt2_ptrs_corpus, opt.tgt_seq_length_trunc,
        "tgt2_ptrs", opt.max_shard_size,
        assoc_iter=src_shard2)
    src_iter = iter(src_shard)
    tgt_iter = iter(tgt_shard)
    src_iter2 = iter(src_shard2)
    tgt_iter2 = iter(tgt_shard2)
    tgt2_ptrs_iter = iter(tgt2_ptrs_shard)

    print("pointers2")
    print(pointers)
    index = 0
    count = 0
    """
    for src, tgt, src2, tgt2, tgt2_ptrs in zip(src_iter, tgt_iter,src_iter2,tgt_iter2,tgt2_ptrs_iter):
        print(src, tgt, src2, tgt2, tgt2_ptrs)
    """
    while not src_shard.hit_end():
        index += 1
        print(count,index)
        dataset = onmt.io.TextDataset(
            fields,
            src_iter,
            tgt_iter,
            src_iter2,
            tgt_iter2,
            tgt2_ptrs_iter,
            src_shard.num_feats,
            tgt_shard.num_feats,
            src_shard2.num_feats,
            tgt_shard2.num_feats,
            tgt2_ptrs_shard.num_feats,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            dynamic_dict=opt.dynamic_dict,
            pointers_file=pointers)

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:d}.pt".format(
            opt.save_data, corpus_type, index)
        print(" * saving %s data shard to %s." % (corpus_type, pt_file))
        check_path(pt_file)
        torch.save(dataset, pt_file)
        ret_list.append(pt_file)

    return ret_list


def build_save_dataset(corpus_type, fields, opt):

    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src1
        tgt_corpus = opt.train_tgt1
        src_corpus2 = opt.train_src2
        tgt_corpus2 = opt.train_tgt2
        pointers = opt.train_ptr
        # pointers = [opt.train_ptr, opt.train_ptr_rsrc]
        tgt2_ptrs_corpus = opt.train_tgt2_ptrs
    else:
        src_corpus = opt.valid_src1
        tgt_corpus = opt.valid_tgt1
        src_corpus2 = opt.valid_src2
        tgt_corpus2 = opt.valid_tgt2
        pointers = None
        tgt2_ptrs_corpus = opt.valid_tgt2_ptrs

    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    print("pointers")
    print(pointers)

    return build_save_text_dataset_in_shards(
        src_corpus, tgt_corpus,src_corpus2, tgt_corpus2,
        tgt2_ptrs_corpus, fields,
        corpus_type,opt,pointers=pointers
    )



def count_features(path, side):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """

    assert side in ['src1', 'src2', 'tgt1', 'tgt2', 'tgt2_ptrs']

    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main(opt):
    # ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    if not(opt.overwrite):
        check_existing_pt_files(opt)

    logger.info("Extracting features...")
    """
    src_nfeats1 = count_features(opt.train_src1, 'src1')
    tgt_nfeats1 = count_features(opt.train_tgt1, 'tgt1')
    src_nfeats2 = count_features(opt.train_src2, 'src2')
    tgt_nfeats2 = count_features(opt.train_tgt2, 'tgt2')
    tgt_ptrs_nfeats2 = count_features(opt.train_tgt2_ptrs, 'tgt2_ptrs')
    """
    src_nfeats1 = onmt.io.get_num_features(
        opt.data_type, opt.train_src1, 'src1')
    tgt_nfeats1 = onmt.io.get_num_features(
        opt.data_type, opt.train_tgt1, 'tgt1')
    src_nfeats2 = onmt.io.get_num_features(
        opt.data_type, opt.train_src2, 'src2')
    tgt_nfeats2 = onmt.io.get_num_features(
        opt.data_type, opt.train_tgt2, 'tgt2')
    tgt_ptrs_nfeats2 = onmt.io.get_num_features(
        opt.data_type, opt.train_tgt2_ptrs, 'tgt2_ptrs')

    logger.info(" * number of source features: %d." % src_nfeats1)
    logger.info(" * number of target features: %d." % tgt_nfeats1)
    logger.info(" * number of source features: %d." % src_nfeats2)
    logger.info(" * number of target features: %d." % tgt_nfeats2)
    logger.info(" * number of target features: %d." % tgt_ptrs_nfeats2)

    logger.info("* Building Fields object...")
    fields = onmt.io.get_fields(opt.data_type, src_nfeats1, tgt_nfeats1)

    logger.info("* Building & saving training data... ")
    train_dataset_files = build_save_dataset('train', fields, opt)

    print("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)

    print("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

"""
    logger.info("* Building `Fields` object...")
    fields = onmt.io.get_fields(opt.data_type, src_nfeats1, tgt_nfeats1)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        tgt_nfeats,
        dynamic_dict=opt.dynamic_dict,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    tgt_reader = inputters.str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset(
        'train', fields, src_reader, tgt_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, src_reader, tgt_reader, opt)
"""


def _get_parser():
    parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    opts.preprocess_opts(parser)
    opts.config_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)