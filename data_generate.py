import random
from os.path import join as pjoin
import numpy as np
from data_simulate import *


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


def refill(batches, fx, batch_size, start=0, end=-1, cur_len=-1, max_seq_len=300, sort_and_shuffle=True):
    line_pairs = []
    linex = fx.readline()
    line_id = 0

    while linex:
        if line_id >= start:
            newline = linex.strip()
            tokens = tokenize(newline)
            if len(tokens) >= 1:
                if cur_len > -1:
                    max_len = min(cur_len, max_seq_len)
                    choice = np.random.choice(max_len, 1)[0] + 1
                else:
                    choice = max_seq_len
                line_pairs.append(tokens[:choice])
        linex = fx.readline()
        line_id += 1
        if line_id == end:
            break

    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e))

    for batch_start in xrange(0, len(line_pairs), batch_size):
        x_batch = line_pairs[batch_start:batch_start + batch_size]
        batches.append(x_batch)

    if sort_and_shuffle:
        random.shuffle(batches)
    return


def refill_var(batches, fx, fy, fc, fp, batch_size, num_wit, start=0, end=-1, cur_len=-1, max_seq_len=300, sort_and_shuffle=True):
    line_pairs = []
    linex = fx.readline()
    if fy:
        liney = fy.readline()
    linec = fc.readline()
    linep = fp.readline()
    line_id = 0

    while linex:
        if line_id >= start:
            x_tokens = tokenize(linex.strip())
            if fy:
                y_tokens = tokenize(liney.strip())
            newc = linec.strip().split('\t')
            newp = linep.strip().split('\t')
            cands = map(lambda ele: [int(s) for s in ele.split()[:num_wit]], newc)
            probs = map(lambda ele: [float(s) for s in ele.split()[:num_wit]], newp)
            if len(x_tokens) >= 1:
                if cur_len > -1:
                    max_len = min(cur_len, max_seq_len)
                    choice = np.random.choice(max_len, 1)[0] + 1
                else:
                    choice = max_seq_len
                if fy:
                    line_pairs.append((x_tokens[:choice], y_tokens[:choice], cands[:choice], probs[:choice]))
                else:
                    line_pairs.append((x_tokens[:choice, x_tokens[:choice], cands[:choice], probs[:choice]]))
        line_id += 1
        linex = fx.readline()
        if fy:
            liney = fy.readline()
        linec = fc.readline()
        linep = fp.readline()
        if line_id == end:
            break

    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e[0]))

    for batch_start in xrange(0, len(line_pairs), batch_size):
        c_batch = [ele[2] for ele in line_pairs[batch_start:batch_start + batch_size]]
        p_batch = [ele[3] for ele in line_pairs[batch_start:batch_start + batch_size]]
        x_batch = [ele[0] for ele in line_pairs[batch_start:batch_start + batch_size]]
        y_batch = [ele[1] for ele in line_pairs[batch_start:batch_start + batch_size]]
        batches.append((x_batch, y_batch, c_batch, p_batch))

    if sort_and_shuffle:
        random.shuffle(batches)
    return


def padded(tokens, num_wit, pad_v=char2id[b'<pad>']):
    len_x = map(lambda x: len(x), tokens)
    maxlen = max(len_x)
    if num_wit > 1:
        padding = np.zeros(num_wit)
        return map(lambda token_list: token_list + [padding] * (maxlen - len(token_list)), tokens)
    else:
        padding = pad_v
        return map(lambda token_list: token_list + [padding] * (maxlen - len(token_list)), tokens)


def pair_iter(file_data, dev, num_wit, num_top=10, batch_size=128,
              data_random="eeg", prior=1, prob_high=0.7,
              prob_in=1.0, prob_back=0.0, flag_generate=True,
              max_seq_len=300,
              cur_len=-2,
              start=0,
              end=-1,
              sort_and_shuffle=False):

        if flag_generate:
            if data_random == 'eeg':
                load_eegs()
            fx = open(pjoin(file_data, dev + '.ids'))
        else:
            filename = '%s.%d' % (data_random, num_top)
            if data_random == 'random':
                if prob_back == 0.0:
                    filename = '%s_%.2f_%.2f_%.2f' % (filename, prob_high, prior, prob_in)
                else:
                    filename = '%s_%.2f_%.2f_%.2f_%.2f' % (filename, prob_high, prior, prob_in, prob_back)
            fc = open(pjoin(file_data, '%s.%s.cand' % (dev, filename)))
            fp = open(pjoin(file_data, '%s.%s.cand' % (dev, filename)))
            if prob_back == 0.0:
                fx = open(pjoin(file_data, dev + '.ids'))
                fy = None
            else:
                fx = open(pjoin(file_data,'%s.%s.ids.x' % (dev, filename)))
                fy = open(pjoin(file_data, '%s.%s.ids.y') % (dev, filename))

        batches = []
        voc_size = len(char2id)
        pad_head = np.zeros(voc_size)
        pad_head[char2id['<sos>']] = 1.

        while True:
            if len(batches) == 0:
                if flag_generate:
                    refill(batches, fx, batch_size, cur_len=cur_len + 1,
                           max_seq_len=max_seq_len,
                           sort_and_shuffle=sort_and_shuffle,
                           start=start, end=end)
                else:
                    refill_var(batches, fx, fy, fc, fp, batch_size, num_wit,
                               cur_len=cur_len + 1, max_seq_len=max_seq_len,
                               sort_and_shuffle=sort_and_shuffle,
                               start=start, end=end)
            if len(batches) == 0:
                fx.close()
                if not flag_generate:
                    fc.close()
                    fp.close()
                break
            if flag_generate:
                x_tokens = batches.pop(0)
                if prob_back > 0.:
                    x_tokens, y_tokens = map(lambda tokenlist: add_backspace(tokenlist, prob_back), x_tokens)
                else:
                    y_tokens = x_tokens
                x_probs = map(lambda tokenlist: [pad_head] + map(lambda ele: generate_simulation(ele,
                                                                                num_wit,
                                                                                top=num_top,
                                                                                prob_high=prob_high,
                                                                                prob_in=prob_in,
                                                                                prior=prior,
                                                                                flag_vec=True,
                                                                                simul=data_random),
                                tokenlist[:-1]),
                          x_tokens)
            else:
                x_tokens, y_tokens, cur_cands, cur_probs = batches.pop(0)
                x_probs = map(lambda cands, probs: [pad_head] +
                                                   map(lambda cand, prob: create_vector(cand, prob),
                                                       cands[:-1], probs[:-1]),
                              cur_cands, cur_probs)
            x_probs_padded = padded(x_probs, len(char2id), 0.0)
            y_padded = padded(y_tokens, 1)
            source_probs = np.transpose(np.array(x_probs_padded), (1, 0, 2))
            source_mask = (np.sum(source_probs, -1) > 0).astype(np.int32)
            target_tokens = np.array(y_padded).T
            yield (source_probs, source_mask, target_tokens)
        return


# ========================================= test ==============================
# for ele in pair_iter_distributed_varlen('test','test.random.10_0.7_3.0_0.8',
#                              2, prior=3,
#                              prob_high=0.7, prob_noncand=0.0,
#                              prob_in=0.8,
#                              batch_size=128,
#                              flag_seq2seq=False,
#                              data_random="random",
#                              sort_and_shuffle=True):
#     print 'ok'
