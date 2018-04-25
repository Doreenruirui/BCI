import string
import random
from os.path import join as pjoin
import numpy as np


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
        # if flag_seq2seq:


            # space_str = ' ' + str(char2id[' ']) + ' '
            # list_words = linex.strip().split(space_str)
            # num_words = len(list_words)
            # randn = np.random.choice(num_words)
            # newline = space_str.join(list_words[:randn])
        # else:
        newline = linex.strip()
        tokens = tokenize(newline)
        if flag_seq2seq:
            randn = np.random.choice(len(tokens))
            tokens = tokens[:randn]
        if len(tokens) >= 2:
            line_pairs.append(tokens)
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
        # return map(lambda token_list: token_list + [padding] * (maxlen - len(token_list)), tokens), \
        #        map(lambda cur_len: [1 if i < cur_len else 0 for i in range(maxlen)], len_x)
    else:
        padding = pad_v
        return map(lambda token_list: token_list + [padding] * (maxlen - len(token_list)), tokens)


def generate_confusion(num_cand, cand_prob):
    list_char = ['_'] + list(string.ascii_lowercase)
    f = open('confusion_matrix', 'w')
    for i in range(len(list_char)):
        cur_char = list_char[i]
        list_new = list_char[:]
        random.shuffle(list_new)
        cur_cand = [ele for ele in list_new[:num_cand] if ele != cur_char]
        if len(cur_cand) < num_cand:
            cur_cand.append(list_new[num_cand])
        cur_prob = [cand_prob] + [(1 - cand_prob) / (num_cand - 1)] * (num_cand - 1)
        f.write(cur_char + '\t' + str(num_cand)
                + '\t' + '\t'.join(cur_cand)
                + '\t' + '\t'.join(map(str, cur_prob)) + '\n')
    f.close()


def get_confusion(cur_char_id, num_cand):
    index = [ele for ele in range(4, len(id2char))]
    random.shuffle(index)
    a = index[:num_cand-1]
    if cur_char_id in a:
        a.append(index[num_cand])
    else:
        a.append(cur_char_id)
    random.shuffle(a)
    return a


def get_confusion_concat(cur_char_id, num_cand):
    index = [ele for ele in range(4, len(id2char))]
    random.shuffle(index)
    a = index[:num_cand-1]
    if cur_char_id in a:
        a.append(index[num_cand])
    else:
        a.append(cur_char_id)
    # random.shuffle(a)
    new_index = np.zeros(len(id2char), dtype=int)
    for item in a:
        new_index[item] = item
    return new_index


def get_confusion_concat_distributed(cur_char_id, num_cand, prior=1, prob_high=0.7, prob_noncand=0.1):
    if cur_char_id == 1:
        prob_vec = np.zeros(len(id2char))
        prob_vec[cur_char_id] = 1.0
        return prob_vec
    elif cur_char_id == 0:
        return np.zeros(len(id2char))
    index = [ele for ele in range(4, len(id2char))]
    random.shuffle(index)
    a = index[:num_cand-1]
    if cur_char_id in a:
        a.append(index[num_cand])
    else:
        a.append(cur_char_id)
    random.shuffle(a)
    prior_vec = np.ones(num_cand)
    prior_vec[num_cand - 1] = prior
    prob_cand = np.random.dirichlet(prior_vec, size=1)[0, :] * (1. - prob_noncand)
    flag_high = np.random.binomial(1, prob_high)
    chosen_id = np.argmax(prob_cand)
    chosen_v = prob_cand[chosen_id]
    if not flag_high:
        cur_id = np.random.choice(5, 1)[0]
        while cur_id == chosen_id:
            cur_id = np.random.choice(5, 1)
        chosen_id = cur_id
        chosen_v = prob_cand[chosen_id]
    cand_prob_vec = []
    for i in range(num_cand):
        if i != chosen_id:
            cand_prob_vec.append(prob_cand[i])
    prob_vec = np.ones(len(id2char)) * (prob_noncand * 1. / (len(id2char) - num_cand))
    cand_id = 0
    for item in a:
        if item == cur_char_id:
            prob_vec[item] = chosen_v
        else:
            prob_vec[item] = cand_prob_vec[cand_id]
            cand_id += 1
    return prob_vec


def load_confusion(file_confusion, num_char):
    global char2id, id2char
    dict_confusion = {}
    for line in file(file_confusion):
        line = line.split('\t')
        cur_char = char2id[line[0]]
        num_char = int(line[1])
        cur_cand = [char2id[ele] for ele in line[2:2 + num_char]]
        cur_prob = map(float, line[2 + num_char:])
        dict_confusion[cur_char] = [cur_cand, cur_prob]
    return dict_confusion


def pair_iter(file_data, num_cand, batch_size=128, concat=0, feed=0, sort_and_shuffle=True):
    fx = open(file_data)
    batches = []

    while True:
        if len(batches) == 0:
            refill(batches, fx, batch_size, sort_and_shuffle)
        if len(batches) == 0:
            break
        x_tokens = batches.pop(0)
        y_tokens = map(lambda tokenlist: tokenlist[1:], x_tokens)
        x_tokens = map(lambda tokenlist: tokenlist[:-1], x_tokens)
        if concat:
            x_tokens = map(lambda tokenlist: [get_confusion_concat(ele, num_cand) for ele in tokenlist], x_tokens)
            if feed:
                x_tokens = map(lambda tokenlist_x, tokenlist_y: map(lambda ele1, ele2: list(ele1) + [ele2],
                                                                    tokenlist_x, [0] + tokenlist_y[:-1]),
                               x_tokens, y_tokens)
                x_padded, x_mask = padded(x_tokens, len(char2id) + 1)
            else:
                x_padded, x_mask = padded(x_tokens, len(char2id))
        else:
            x_tokens = map(lambda tokenlist: [get_confusion(ele, num_cand) for ele in tokenlist], x_tokens)
            if feed:
                x_tokens = map(lambda tokenlist_x, tokenlist_y: map(lambda ele1, ele2: ele1 + [ele2],
                                                                    tokenlist_x, [0] + tokenlist_y[1:]),
                               x_tokens, y_tokens)
                x_padded, x_mask = padded(x_tokens, num_cand + 1)
            else:
                x_padded, x_mask = padded(x_tokens, num_cand)
        y_padded = padded(y_tokens, 1)
        source_tokens = np.transpose(np.array(x_padded), (1, 0, 2))
        source_mask = np.array(x_mask).astype(np.int32).T
        target_tokens = np.array(y_padded).T
        yield (source_tokens, source_mask, target_tokens)

    return


def pair_iter_word(file_data, num_cand, batch_size=128, concat=0, feed=0, sort_and_shuffle=True):
    fx = open(file_data)
    fy = open(file_data + '.word')
    batches = []

    while True:
        if len(batches) == 0:
            refill_word(batches, fx, fy, batch_size, sort_and_shuffle)
        if len(batches) == 0:
            break
        x_tokens, y_tokens_word = batches.pop(0)
        y_tokens = x_tokens
        if concat:
            x_tokens = map(lambda tokenlist: [get_confusion_concat(ele, num_cand) for ele in tokenlist], x_tokens)
            if feed:
                x_tokens = map(lambda tokenlist_x, tokenlist_y: map(lambda ele1, ele2: list(ele1) + [ele2],
                                                                    tokenlist_x, [0] + tokenlist_y[:-1]),
                               x_tokens, y_tokens)
                x_padded, x_mask = padded(x_tokens, len(char2id) + 1)
            else:
                x_padded, x_mask = padded(x_tokens, len(char2id))
        else:
            x_tokens = map(lambda tokenlist: [get_confusion(ele, num_cand) for ele in tokenlist], x_tokens)
            if feed:
                x_tokens = map(lambda tokenlist_x, tokenlist_y: map(lambda ele1, ele2: ele1 + [ele2],
                                                                    tokenlist_x, [0] + tokenlist_y[1:]),
                               x_tokens, y_tokens)
                x_padded, x_mask = padded(x_tokens, num_cand + 1)
            else:
                x_padded, x_mask = padded(x_tokens, num_cand)
        y_padded = padded(y_tokens, 1)
        y_padded_word = padded(y_tokens_word, 1)
        source_tokens = np.transpose(np.array(x_padded), (1, 0, 2))
        source_mask = np.array(x_mask).astype(np.int32).T
        target_tokens = np.array(y_padded).T
        target_tokens_word = np.array(y_padded_word).T
        yield (source_tokens, source_mask, target_tokens, target_tokens_word)

    return


def pair_iter_distributed(file_data, num_cand, prior=1, prob_high=0.7, prob_noncand=0.0, batch_size=128, feed=0, flag_seq2seq=False, sort_and_shuffle=True):
    fx = open(file_data)
    batches = []

    while True:
        if len(batches) == 0:
            refill(batches, fx, batch_size, flag_seq2seq, sort_and_shuffle)
        if len(batches) == 0:
            break
        x_tokens = batches.pop(0)
        y_tokens = x_tokens
        x_tokens = map(lambda tokenlist: [char2id['<sos>']] + tokenlist[:-1] + [char2id['<pad>']], x_tokens)
        x_probs = map(lambda tokenlist: map(lambda ele:
                                            get_confusion_concat_distributed(ele, num_cand,
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
        # source_mask = np.array(x_mask).astype(np.int32).T
        source_mask = (np.sum(source_probs, -1) > 0).astype(np.int32)
        target_tokens = np.array(y_padded).T
        yield (source_probs, source_mask, target_tokens)
    return


def pair_iter_word_distributed(file_data, num_cand, prior=1, prob_high=0.7, prob_noncand=0.0, batch_size=128, feed=0, sort_and_shuffle=True):
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
                                            get_confusion_concat_distributed(ele, num_cand,
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


# def test():
#     generate_confusion(10, 0.1)
#     for x in pair_iter_distributed('/home/rui/Dataset/BCI/test.ids.x', 5, prior=3):
#         print 1
#
#
# test()

