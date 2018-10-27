import sys
import numpy as np
import string

file_name = sys.argv[1]
noisy_level = float(sys.argv[2])

def generate(filename, noisy_rate):
    with open(filename + '_' + str(noisy_level), 'w') as f_out:
        with open(filename, 'r') as f_in:
            for line in f_in:
                line = line.strip().lower()
                items = list(line)
                len_line = len(line)
                index = np.arange(len_line)
                np.random.shuffle(index)
                nchoice = int(np.ceil(len_line * noisy_rate))
                chose_id = index[:nchoice]
                for cid in chose_id:
                    cur_char = line[cid]
                    cands = [' '] + list(string.ascii_lowercase)
                    cands.remove(cur_char)
                    cand_id = np.random.choice(26, 1)
                    rep = cands[cand_id[0]]
                    items[cid] = rep
                f_out.write(''.join(items) + '\n')
        f_in.close()
    f_out.close()

generate(file_name, noisy_level)