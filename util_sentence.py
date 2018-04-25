import string
import re
from nltk.tokenize import sent_tokenize
import sys
import numpy as np
from os.path import join as pjoin


id2char = [b'<pad>', b'<sos>', b'<eos>', b' '] + list(string.ascii_lowercase)
char2id = {k: v for v, k in enumerate(id2char)}


def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def char_tokenize(sent):
    return [char2id[ele] for ele in sent.strip() if ele in char2id]


def vocabulary(path_data, file_data, max_vocabulary_size):
    dict_words = {}
    for line in file(pjoin(path_data, file_data +'.line')):
        line = line.lower().strip()
        for word in line.split(' '):
            dict_words[word] = dict_words.get(word, 0) + 1
    vocab = sorted(dict_words, key=dict_words.get, reverse=True)
    if len(vocab) > max_vocabulary_size - 6:
        vocab = vocab[:max_vocabulary_size - 6]
    with open(pjoin(path_data, 'vocab'), 'w') as f_out:
        f_out.write('<PAD>\n')
        f_out.write('<SOS>\n')
        f_out.write('<EOS>\n')
        f_out.write(' \n')
        for w in vocab:
           f_out.write(w + b"\n")
        f_out.write('<UNK>\n')


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


def tokenize(path_data, file_data):
    with open(pjoin(path_data, file_data + '.ids'), 'w') as f_out:
        for line in file(pjoin(path_data, file_data + '.line')):
            line = line.strip().lower()
            if len(line) > 1:
                cur_token = char_tokenize(line)
                f_out.write(' '.join(map(str, cur_token)) + '\n')


def word_tokenize(path_data, file_data):
    vocab, rev_vocab = load_vocabulary(path_data)
    with open(pjoin(path_data, file_data + '.ids.word'), 'w') as f_out:
        for line in file(pjoin(path_data, file_data +  '.line')):
            line = line.strip().lower()
            if len(line) > 1:
                list_words = line.split(' ')
                if list_words[0] in vocab:
                    cur_word_id = vocab[list_words[0]]
                else:
                    cur_word_id = vocab['<UNK>']
                cur_line = ' '.join(map(str,[cur_word_id] * len(list_words[0])))
                for word in list_words[1:]:
                    if word in vocab:
                        cur_word_id = vocab[word]
                    else:
                        cur_word_id = vocab['<UNK>']
                    cur_line += ' ' + str(vocab[' ']) + ' ' + ' '.join(map(str, [cur_word_id] * len(word)))
                f_out.write(cur_line + '\n')


def reprocess_file(path_data, file_data, max_num_words):
    num_word = 0
    num_sen = 0
    list_num_word = []
    with open(pjoin(path_data, file_data + '.line'), 'w')  as f_out:
        for line in file(pjoin(path_data, file_data)):
            list_sent = sent_tokenize(line.strip())
            for sent in list_sent:
                sent = remove_nonascii(sent)
                sent = re.sub(' +', ' ', re.sub('-', ' ', sent))
                # sent = re.sub(' +', ' ', ''.join([ele if ele.isalpha() else ' ' for ele in sent]))
                list_words = [''.join([c if c.isalpha() else '' for c in ele])
                              for ele in sent.strip().split(' ')]
                list_words = [ele for ele in list_words if len(ele.strip()) > 0]
                len_line = len(list_words)
                f_out.write(' '.join(list_words[:max_num_words]) + '\n')
                num_word += len_line
                list_num_word.append(len_line)
                num_sen += 1
    print num_word * 1. / num_sen
    print np.mean(list_num_word)
    print max(list_num_word), min(list_num_word)



reprocess_file(sys.argv[1], sys.argv[2], 20)
vocabulary(sys.argv[1], sys.argv[2], 50000)
tokenize(sys.argv[1], sys.argv[2])
word_tokenize(sys.argv[1], sys.argv[2])
