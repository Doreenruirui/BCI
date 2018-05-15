import string
import random
from os.path import join as pjoin
import numpy as np
from data_simulate_eeg import generate_direchlet, generate_eeg, generate_clean, load_eegs


id2char = [b'<pad>', b'<sos>', b'<eos>', b' '] + list(string.ascii_lowercase)
char2id = {k: v for v, k in enumerate(id2char)}


def tokenize(string):
  return [int(s) for s in string.split()]


def load_vocabulary(path_data):
    vocab = {}
    rev_vocab = {}
    line_id = 0
    for line in file(pjoin(path_data, 'vocab')):
        word = line.strip('\n')
        vocab[word] = line_id
        rev_vocab[line_id] = word
        line_id += 1
    return vocab, rev_vocab


def refill(batches, fx, batch_size, flag_seq2seq=False, sort_and_shuffle=True):
    line_pairs = []
    linex = fx.readline()
    while linex:
        newline = linex.strip()
        tokens = tokenize(newline)
        if flag_seq2seq:
            randn = np.random.choice(len(tokens))
            tokens = tokens[:randn]
        if len(tokens) >= 2:
            line_pairs.append(tokens[:200])
        linex = fx.readline()

    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e))

    for batch_start in xrange(0, len(line_pairs), batch_size):
        x_batch = line_pairs[batch_start:batch_start + batch_size]
        batches.append(x_batch)

    if sort_and_shuffle:
        random.shuffle(batches)
    return


def refill_word(batches, fx, fy, batch_size, sort_and_shuffle=True):
    line_pairs = []
    linex = fx.readline()
    liney = fy.readline()

    while linex and liney:
        tokens_x = tokenize(linex.strip())
        tokens_y = tokenize(liney.strip())
        line_pairs.append([tokens_x, tokens_y])
        linex = fx.readline()
        liney = fy.readline()

    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e[0]))

    for batch_start in xrange(0, len(line_pairs), batch_size):
        x_batch, y_batch = zip(*line_pairs[batch_start:batch_start+batch_size])
        batches.append((x_batch, y_batch))

    if sort_and_shuffle:
        random.shuffle(batches)
    return


def padded(tokens, num_cand, pad_v=char2id[b'<pad>']):
    len_x = map(lambda x: len(x), tokens)
    maxlen = max(len_x)
    if num_cand > 1:
        padding = [pad_v] * num_cand
        return map(lambda token_list: token_list + [padding] * (maxlen - len(token_list)), tokens)
    else:
        padding = pad_v
        return map(lambda token_list: token_list + [padding] * (maxlen - len(token_list)), tokens)


def pair_iter_distributed(file_data, num_cand, prior=1,
                          prob_high=0.7, prob_noncand=0.0,
                          batch_size=128, feed=0,
                          flag_seq2seq=False,
                          data_random="eeg",
                          sort_and_shuffle=True):
    fx = open(file_data)
    batches = []
    load_eegs(nbest=num_cand)

    while True:
        if len(batches) == 0:
            refill(batches, fx, batch_size, flag_seq2seq, sort_and_shuffle)
        if len(batches) == 0:
            break
        x_tokens = batches.pop(0)
        y_tokens = x_tokens
        x_tokens = map(lambda tokenlist: [char2id['<sos>']] +
                                         tokenlist[:-1] +
                                         [char2id['<pad>']], x_tokens)
        if data_random == "clean":
            x_probs = map(lambda tokenlist: map(lambda ele:
                                                generate_clean(ele)
                                                ,tokenlist),
                          x_tokens)
        elif data_random == "eeg":
            x_probs = map(lambda tokenlist: map(lambda ele:
                                                generate_eeg(ele, num_cand),
                                                tokenlist),
                          x_tokens)
        else:
            x_probs = map(lambda tokenlist: map(lambda ele:
                                                generate_direchlet(ele, num_cand,
                                                                   prior=prior,
                                                                   prob_high=prob_high,
                                                                   prob_noncand=prob_noncand),
                                                    tokenlist),
                              x_tokens)
        if feed:
            x_probs = map(lambda problist_x: map(lambda ele1: list(ele1) + [1.],
                                                       problist_x),
                                x_probs)
            x_probs_padded = padded(x_probs, len(char2id) + 1, 0.0)
        else:
            x_probs_padded = padded(x_probs, len(char2id), 0.0)
        y_padded = padded(y_tokens, 1)
        y_padded = map(lambda tokenlist: [char2id['<sos>']] + tokenlist, y_padded)
        source_probs = np.transpose(np.array(x_probs_padded), (1, 0, 2))
        source_mask = (np.sum(source_probs, -1) > 0).astype(np.int32)
        target_tokens = np.array(y_padded).T
        yield (source_probs, source_mask, target_tokens)
    return


def pair_iter_word_distributed(file_data, num_cand, prior=1,
                               prob_high=0.7, prob_noncand=0.0,
                               batch_size=128, feed=0, sort_and_shuffle=True):
    fx = open(file_data)
    fy = open(file_data + '.word')
    batches = []

    while True:
        if len(batches) == 0:
            refill_word(batches, fx, fy, batch_size, sort_and_shuffle)
        if len(batches) == 0:
            break
        x_tokens, y_tokens_word = batches.pop(0)
        # y_tokens = x_tokens
        y_tokens = map(lambda tokenlist: tokenlist[1:], x_tokens)
        x_tokens = map(lambda tokenlist: tokenlist[:-1], x_tokens)
        y_tokens_word = map(lambda tokens: tokens[1:], y_tokens_word)
        x_probs = map(lambda tokenlist: map(lambda ele:
                                            generate_direchlet(ele, num_cand,
                                                               prior=prior,
                                                               prob_high=prob_high,
                                                               prob_noncand=prob_noncand),
                                            tokenlist),
                      x_tokens)
        if feed:
            x_probs = map(lambda problist_x: map(lambda ele1: list(ele1) + [1.],
                                                       problist_x),
                                x_probs)
            x_probs_padded, x_mask = padded(x_probs, len(char2id) + 1, 0.0)
        else:
            x_probs_padded, x_mask = padded(x_probs, len(char2id), 0.0)
        y_padded = padded(y_tokens, 1)
        source_probs = np.transpose(np.array(x_probs_padded), (1, 0, 2))
        source_mask = np.array(x_mask).astype(np.int32).T
        target_tokens = np.array(y_padded).T
        y_padded_word = padded(y_tokens_word, 1)
        target_tokens_word = np.array(y_padded_word).T
        yield (source_probs, source_mask, target_tokens, target_tokens_word)
    return

