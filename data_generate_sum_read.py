import random
from os.path import join as pjoin
from data_simulate import *
import datetime




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


def refill(batches, fx, batch_size, num_wit, start=0, end=-1, max_seq_len=300, sort_and_shuffle=True):
    line_pairs = []
    linex = fx.readline()
    line_id = 0

    while linex:
        if line_id >= start:
            tokens = list(map(int, linex.strip().split()))
            if len(tokens) >= 1:
                line_pairs.append(tokens[:max_seq_len])
        linex = fx.readline()
        line_id += 1
        if line_id == end:
            break
    # line_pairs = pool.map(generate_sentence, line_pairs)

    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e))

    for batch_start in range(0, len(line_pairs), batch_size):
        x_batch = line_pairs[batch_start:batch_start + batch_size]
        batches.append(x_batch)

    if sort_and_shuffle:
        random.shuffle(batches)

def refill_var(batches, fx, fc, fp, batch_size, num_wit,  max_seq_len=300, sort_and_shuffle=True, end=-1):
    line_pairs = []
    linex, linec, linep = fx.readline(), fc.readline(), fp.readline()

    print("read_data :", datetime.datetime.now())
    print("num_wit :", num_wit)
    while linex and linep and linec:
        sen = list(map(int, linex.strip('\n').split(' ')))
        newc = linec.strip().split('\t')
        newp = linep.strip().split('\t')
        cands =  [[char2id['<sos>']] + [0] * (num_wit - 1)]  + list(map(lambda ele: [int(s) for s in ele.split(' ')[:num_wit]], newc[:-1]))
        #print(linep)
        probs = [[1.] + [0.] * (num_wit - 1)]  + list(map(lambda ele: [float(s) for s in ele.split(' ')[:num_wit]], newp[:-1]))
       
        probs = list(map(lambda ele: [e/sum(ele) for e in ele], probs))
        line_pairs.append((sen[:max_seq_len], cands[:max_seq_len], probs[:max_seq_len]))
        if len(line_pairs) == end:
            break
        linex, linec, linep = fx.readline(), fc.readline(), fp.readline()
    # print(sum(line_pairs[-1][-1]))
    print("shuffle1 :", datetime.datetime.now())
    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e[0]))

    print("generate :", datetime.datetime.now())
    for batch_start in range(0, len(line_pairs), batch_size):
        x_batch, c_batch, p_batch = zip(*line_pairs[batch_start:batch_start+batch_size])
        batches.append((x_batch, c_batch, p_batch))

    print("shuffle2 :", datetime.datetime.now())
    if sort_and_shuffle:
        random.shuffle(batches)
    return


def pair_iter(file_data, dev, num_wit, batch_size=128,
              max_seq_len=300,
              prob_vec='0.7_0.1',
              prior_vec='3_1',
              flag_generate=False, sort_and_shuffle=False):
    fx = open(pjoin(file_data, '0.0', dev + '.ids'))
    if not flag_generate:
        fc = open(pjoin(file_data, '0.0', '%s_prior_%s_prob_%s.cand' % (dev, prior_vec, prob_vec)))
        fp = open(pjoin(file_data, '0.5', '%s.prob' % dev))
    batches = []
    while True:
        if len(batches) == 0:
            if flag_generate:
                refill(batches, fx, batch_size, num_wit,
                       max_seq_len=300, sort_and_shuffle=sort_and_shuffle)
            else:
                refill_var(batches, fx, fc, fp, batch_size, num_wit,
                           sort_and_shuffle=sort_and_shuffle, end=16 * 10000, max_seq_len=max_seq_len)
        if len(batches) == 0:
            fx.close()
            if not flag_generate:
                fp.close()
                fc.close()
            break
        y_tokens, x_cands, x_probs = batches.pop(0)
        x_probs_padded = padded(x_probs, num_wit, 0.0)
        x_cands_padded = padded(x_cands, num_wit, 0)
        # print(x_cands_padded)
        source_probs = np.transpose(np.array(x_probs_padded), (1, 0, 2))
        source_cands = np.transpose(np.array(x_cands_padded), (1, 0, 2))
        y_padded = padded(y_tokens, 0)
        source_mask = (np.sum(source_probs, -1) > 0).astype(np.int32)
        target_tokens = np.array(y_padded).T
        yield (source_cands, source_probs, source_mask, target_tokens)
    return


def padded(tokens, num_wit, pad_v=char2id['<pad>']):
    len_x = list(map(lambda x: len(x), tokens))
    maxlen = max(len_x)
    if num_wit >= 1:
        padding = [pad_v] * num_wit
        return list(map(lambda token_list: token_list + [padding] * (maxlen - len(token_list)), tokens))
    else:
        padding = pad_v
        return list(map(lambda token_list: token_list + [padding] * (maxlen - len(token_list)), tokens))


def load_data(batches, file_data, dev, batch_size=128,
              num_wit=2,
              max_seq_len=300,
              prob_vec='0.7_0.1',
              prior_vec='3_1',
              flag_generate=False, sort_and_shuffle=False):
    fx = open(pjoin(file_data, '0.0', dev + '.ids'))
    if not flag_generate:
        fc = open(pjoin(file_data, '0.0', '%s_prior_%s_prob_%s.cand' % (dev, prior_vec, prob_vec)))
        fp = open(pjoin(file_data, '0.5', '%s.prob' % dev))
    print (pjoin(file_data, '0.0', '%s_prior_%s_prob_%s.cand' % (dev, prior_vec, prob_vec)))
    if flag_generate:
        refill(batches, fx, batch_size, num_wit,
               max_seq_len=max_seq_len, sort_and_shuffle=sort_and_shuffle)
    else:
        refill_var(batches, fx, fc, fp, batch_size, num_wit, max_seq_len=max_seq_len,
                   sort_and_shuffle=sort_and_shuffle)
    fx.close()
    if not flag_generate:
        fp.close()
        fc.close()


def iter_data(batch, num_wit):
    y_tokens, x_cands, x_probs = batch
    x_probs_padded = padded(x_probs, num_wit, 0.0)
    x_cands_padded = padded(x_cands, num_wit, 0)
    # print(x_cands_padded)
    source_probs = np.transpose(np.array(x_probs_padded), (1, 0, 2))
    source_cands = np.transpose(np.array(x_cands_padded), (1, 0, 2))
    y_padded = padded(y_tokens, 0)
    source_mask = (np.sum(source_probs, -1) > 0).astype(np.int32)
    target_tokens = np.array(y_padded).T
    return (source_cands, source_probs, source_mask, target_tokens)


def data_generate(pool, batch, num_wit):
    x_tokens = batch
    res = pool.map(generate_sentence, x_tokens)
    x_cands = [ele[0][:-1] for ele in res]
    x_probs = [ele[1][:-1] for ele in res]
    x_probs_padded = padded(x_probs, num_wit, 0.0)
    x_cands_padded = padded(x_cands, num_wit, 0)
    source_probs = np.transpose(np.array(x_probs_padded), (1, 0, 2))
    source_cands = np.transpose(np.array(x_cands_padded), (1, 0, 2))
    y_padded = padded(x_tokens, 0)
    source_mask = (np.sum(source_probs, -1) > 0).astype(np.int32)
    target_tokens = np.array(y_padded).T
    return (source_cands, source_probs, source_mask, target_tokens)

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
