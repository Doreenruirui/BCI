import random
from os.path import join as pjoin
from data_simulate import *
import datetime


def tokenize(string):
  return [int(s) for s in string.split()]


def load_vocabulary(path_data):
    vocab = {}
    rev_vocab = {}
    line_id = 0
    with open(pjoin(path_data, 'vocab')) as f_in:
        for line in f_in:
            word = line.strip('\n')
            vocab[word] = line_id
            rev_vocab[line_id] = word
            line_id += 1
    return vocab, rev_vocab


def refill(batches, fx, batch_size, start=0, end=-1, max_seq_len=300, sort_and_shuffle=True):
    line_pairs = []
    linex = fx.readline()
    line_id = 0

    while linex:
        if line_id >= start:
            newline = linex.strip()
            tokens = tokenize(newline)
            if len(tokens) >= 1:
                tok_x = [char2id['<sos>']] + tokens[:max_seq_len][:-1]
                tok_y = tokens[:max_seq_len]
                line_pairs.append((tok_x, tok_y))
        linex = fx.readline()
        line_id += 1
        if line_id == end:
            break

    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e[0]))

    for batch_start in range(0, len(line_pairs), batch_size):
        x_batch = [ele[0] for ele in line_pairs[batch_start:batch_start + batch_size]]
        y_batch = [ele[1] for ele in line_pairs[batch_start:batch_start + batch_size]]
        batches.append((x_batch, y_batch))

    if sort_and_shuffle:
        random.shuffle(batches)
    return

def refill_noisy(batches, fx, fy, batch_size, start=0, end=-1, max_seq_len=300, sort_and_shuffle=True):
    line_pairs = []
    linex = fx.readline()
    liney = fy.readline()
    line_id = 0

    while linex:
        if line_id >= start:
            newline_x = linex.strip()
            newline_y = liney.strip()
            tokens_x = tokenize(newline_x)
            tokens_y = tokenize(newline_y)
            if len(tokens_x) >= 1:
                tok_x = [char2id['<sos>']] + tokens_x[:max_seq_len][:-1]
                tok_y = tokens_y[:max_seq_len]
                line_pairs.append((tok_x, tok_y))
        linex = fx.readline()
        liney = fy.readline()
        line_id += 1
        if line_id == end:
            break

    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e[0]))

    for batch_start in range(0, len(line_pairs), batch_size):
        x_batch = [ele[0] for ele in line_pairs[batch_start:batch_start + batch_size]]
        y_batch = [ele[1] for ele in line_pairs[batch_start:batch_start + batch_size]]
        batches.append((x_batch, y_batch))

    if sort_and_shuffle:
        random.shuffle(batches)
    return


def padded(tokens, pad_v=char2id['<pad>']):
    len_x = list(map(lambda x: len(x), tokens))
    maxlen = max(len_x)
    return list(map(lambda token_list: token_list + [pad_v] * (maxlen - len(token_list)), tokens))

def load_data(batches, file_data, dev, batch_size=128, max_seq_len=300,
                    prob_high=1.0, start=0, end=-1, sort_and_shuffle=False):
    if prob_high == 1.0:
        fx = open(pjoin(file_data, '0.0', dev + '.ids'))
        refill(batches, fx, batch_size, max_seq_len=max_seq_len,
               sort_and_shuffle=sort_and_shuffle, start=start, end=end)
    else:
        fy = open(pjoin(file_data, '0.0', dev + '.ids'))
        fx = open(pjoin(file_data, '%.1f' % (1 - prob_high), dev + '.ids'))
        refill_noisy(batches, fx, fy, batch_size, max_seq_len=max_seq_len,
               sort_and_shuffle=sort_and_shuffle, start=start, end=end)
        fy.close()
    fx.close()

def iter_data(batch):
    x_tokens, y_tokens = batch
    x_pad = padded(x_tokens, 0)
    y_pad = padded(y_tokens, 0)
    source_tokens = np.transpose(np.array(x_pad), (1, 0))
    source_mask = (source_tokens > 0).astype(np.int32)
    target_tokens = np.array(y_pad).T
    return (source_tokens, source_mask, target_tokens)


