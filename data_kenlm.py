import string
import sys

def process_file(filename):
    with open(filename, 'r') as f_in:
        with open(filename + '.kenlm', 'w') as f_out:
            for line in f_in:
                line = ' '.join(line.strip().lower().split())
                items = [ele if ele != ' ' else '#' for ele in line]
                f_out.write(' '.join(items) + '\n')


def remove_empty(filename):
    voc = list(string.ascii_lowercase) + [' ']
    char2id = {val:k for k, val in enumerate(voc)}
    with open(filename, 'r') as f_in:
        with open(filename + '.new', 'w') as f_out:
            for line in f_in:
                line = ' '.join(line.strip().lower().split())
                items = [ele for ele in line if ele in char2id]
                if len(items) > 1:
                    f_out.write(''.join(items) + '\n')


filename = sys.argv[1]
#remove_empty(filename)
#process_file(filename + '.new')
process_file(filename)
