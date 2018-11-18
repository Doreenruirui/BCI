import random
from os.path import join as pjoin
from data_simulate import *
import datetime
from multiprocessing import Pool


def get_bin(prob_vec):
    prob_vec = list(map(float, prob_vec.split('_')))
    num_wit = len(prob_vec)
    bin = [[] for _ in range(num_wit)]
    bin[0].append(0)
    bin_no = 0
    for i in range(num_wit):
        cur_prob = prob_vec[i]
        if i == 0:
            num_bin = int(np.around((cur_prob - 0.5) / 0.025))
        else:
            num_bin = int(np.around(cur_prob / 0.025))
        bin[i] += [ele + bin_no + 1 for ele in range(num_bin)]
        bin_no += num_bin
    dict_bin = {}
    for i in range(num_wit):
        for ele in bin[i]:
            dict_bin[ele] = i
    print(dict_bin)
    return dict_bin

def initialize(prior, prob):
    global prior_vec, prob_vec
    prior_vec = prior
    prob_vec = prob

def initialize_fix(arg_bin, arg_nwit, arg_max_len):
    global bin, nwit, max_len
    bin = arg_bin
    nwit = arg_nwit
    max_len = arg_max_len


def generate_sentence(sentence):
    global prior_vec, prob_vec
    res = list(map(lambda ele: generate_dirichlet_vector(ele,
                                                         prior=prior_vec,
                                                         prob_vec=prob_vec),
                   sentence))
    cands = '\t'.join([' '.join(list(map(str, ele[0]))) for ele in res])
    probs = '\t'.join([' '.join(list(map(str, ele[1]))) for ele in res])
    return cands, probs


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


def refill(batches, fx, batch_size, start=0, end=-1, max_seq_len=300, prior=None, prob=None, sort_and_shuffle=True):
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

    pool = Pool(processes=50, initializer=initialize(prior, prob))
    line_pairs = pool.map(generate_sentence, line_pairs)

    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e))

    for batch_start in range(0, len(line_pairs), batch_size):
        x_batch = line_pairs[batch_start:batch_start + batch_size]
        batches.append(x_batch)

    if sort_and_shuffle:
        random.shuffle(batches)
    return


def process_line(paras):
    global bin, nwit, max_len
    linex, linec, linep, liner = paras
    sen = list(map(int, linex.strip('\n').split(' ')))
    newc = linec.strip().split('\t')
    newp = linep.strip().split('\t')
    newr = liner.strip().split('\t')
    cands = list(map(lambda ele: [int(s) for s in ele.split()], newc))
    probs = list(map(lambda ele: [float(s) for s in ele.split()], newp))
    ranks = list(map(int, newr))
    res_cand = []
    res_prob = []
    for tok, rank, cand, prob in zip(sen, ranks, cands, probs):
        if rank in bin:
            cur_bin = bin[rank]
            new_cand = [ele for ele in cand]
            new_cand[rank] = new_cand[cur_bin]
            new_cand[cur_bin] = tok
            new_cand = new_cand[:nwit]
        else:
            new_cand = cand[:nwit]
        sum_prob = sum(prob[:nwit])
        new_prob = [ele / sum_prob for ele in prob[:nwit]]
        res_cand.append(new_cand)
        res_prob.append(new_prob)
    res_cand = [[char2id['<sos>']] + [0] * (nwit - 1)] + res_cand[:-1]
    res_prob = [[1.] + [0.] * (nwit - 1)] + res_prob[:-1]
    return sen[:max_len], res_cand[:max_len], res_prob[:max_len]


def refill_var(batches, fx, fc, fp, fr, dict_bin, batch_size, num_wit, start=0, end=-1,
               max_seq_len=300, sort_and_shuffle=True):
    line_pairs = []
    lines = []
    linex = fx.readline()
    linec = fc.readline()
    linep = fp.readline()
    liner = fr.readline()
    line_id = 0
    # pad_head = [char2id['<sos>']] + [0] * (num_wit - 1)
    # pad_prob = [1.] + [0.] * (num_wit - 1)
    pool = Pool(processes=50, initializer=initialize_fix(dict_bin, num_wit, max_seq_len))
    print("read_data :", datetime.datetime.now())
    while linex and linep and linec and liner:
        if line_id >= start:
            lines.append((linex, linec, linep, liner))
            if (line_id + 1) % 1000 == 0:
                res = pool.map(process_line, lines)
                line_pairs += res
                lines = []
        linex = fx.readline()
        linec = fc.readline()
        linep = fp.readline()
        liner = fr.readline()
        line_id += 1
        if line_id == end:
            break
    if len(lines) > 0:
        res = pool.map(process_line, lines)
        line_pairs += res
    pool.close()
    # num_group = np.zeros(5)
    # total = 0
    # nwrong = 0
    # for sen, cand, prob in line_pairs:
    #     for i in range(1, len(sen)):
    #         if sen[i - 1] == cand[i][0]:
    #             num_group[0] += 1
    #         elif sen[i - 1] == cand[i][1]:
    #             num_group[1] += 1
    #         # elif sen[i - 1] == cand[i][2]:
    #         #     num_group[2] += 1
    #         else:
    #             num_group[3] += 1
    #         if len(np.unique(cand[i])) < num_wit:
    #             nwrong += 1
    #     total += len(sen) - 1
    # print(num_group, total, num_group/total, nwrong)

    print("shuffle1 :", datetime.datetime.now())
    if sort_and_shuffle:
        random.shuffle(line_pairs)
        line_pairs = sorted(line_pairs, key=lambda e:len(e[0]))

    print("generate :", datetime.datetime.now())
    for batch_start in range(0, len(line_pairs), batch_size):
        c_batch = [ele[1] for ele in line_pairs[batch_start:batch_start + batch_size]]
        p_batch = [ele[2] for ele in line_pairs[batch_start:batch_start + batch_size]]
        x_batch = [ele[0] for ele in line_pairs[batch_start:batch_start + batch_size]]
        batches.append((x_batch, c_batch, p_batch))

    print("shuffle2 :", datetime.datetime.now())
    if sort_and_shuffle:
        random.shuffle(batches)
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


def load_data(batches, file_data, dev, num_wit, dict_bin, batch_size=128,
              prior_vec='3_1', prob_vec='0.8_0.1',
              max_seq_len=300, start=0, end=-1,
              flag_generate=False, sort_and_shuffle=False):
    fx = open(pjoin(file_data, '0.0', dev + '.ids'))
    if flag_generate:
        refill(batches, fx, batch_size, start=0, end=-1,
               prior=prior_vec, prob=prob_vec,
               max_seq_len=300, sort_and_shuffle=sort_and_shuffle)
    else:
        fc = open(pjoin(file_data, '0.5', '%s.cand' % dev))
        fp = open(pjoin(file_data, '0.5', '%s.prob' % dev))
        fr = open(pjoin(file_data, '0.5', '%s.rank' % dev))
        refill_var(batches, fx, fc, fp, fr, dict_bin, batch_size, num_wit,
                   max_seq_len=max_seq_len,
                   sort_and_shuffle=sort_and_shuffle,
                   start=start, end=end)

        fc.close()
        fp.close()
    fx.close()


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
