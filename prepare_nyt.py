import string
import re
from os.path import join, exists
from os import listdir
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool


folder_data = "/gss_gpfs_scratch/dong.r/Dataset/unprocessed/NYT"


def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def process_file(filename):
    cur_file = join(folder_data, filename)
    with open(join(folder_data, filename + '.sen'), 'w') as f_:
        for line in file(cur_file):
            line = remove_nonascii(line.strip())
            if len([ele for ele in line.strip() if ele.isalpha()]) < 40:
                continue
            else:
                list_sent = sent_tokenize(line.strip())
                for sent in list_sent:
                    sent = re.sub(' +', ' ', re.sub('-', ' ', sent))
                    list_words = [''.join([c if c.isalpha() else '' for c in ele])
                                  for ele in sent.strip().split(' ')]
                    list_words = [ele for ele in list_words if len(ele.strip()) > 0]
                    f_.write(' '.join(list_words) + '\n')


def process_folder():
    list_fn = [ele for ele in listdir(folder_data)  if ele.endswith('.txt')]
    pool = Pool(10)
    pool.map(process_file, list_fn)


process_folder()
