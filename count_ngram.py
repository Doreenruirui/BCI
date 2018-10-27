import sys


def count(filename, ngram):
    res = {}
    with open(filename) as f_in:
        for line in f_in:
            line = line.strip()
            line = '_' + line
            len_line = len(line)
            if len_line < ngram:
                continue
            for i in range(len_line - ngram + 1):
                cur_ngram = line[i:i+ngram]
                res[cur_ngram] = res.get(cur_ngram, 0) + 1
    sorted_res = sorted(res, key=res.get, reverse=True)
    with open('%s.%dgram' % (filename, ngram), 'w') as f_out:
        for ele in sorted_res:
            f_out.write(ele + '\t' + str(res[ele]) + '\n')

# def load_ngram(filename):

count(sys.argv[1], int(sys.argv[2]))

