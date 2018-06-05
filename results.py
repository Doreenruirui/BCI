import sys
from os.path import join as pjoin

def evaluate_mrr():
    mrr = 0.
    num_char = 0
    for i in range(10):
        start = i * 50
        end = (i + 1) * 50
        for line in file(pjoin(sys.argv[1], 'mrr.%d_%d' % (start, end))):
            cur_line = map(float, line.strip().split('\t'))
            mrr += sum(cur_line)
            num_char += len(cur_line)
    print 'recall', mrr / num_char


def evaluate_recall():
    recall = 0.
    num_char = 0
    for i in range(10):
        start = i * 50
        end = (i + 1) * 50
        for line in file(pjoin(sys.argv[1], 'recall.%d_%d' % (start, end))):
            cur_line = map(float, line.strip().split('\t'))
            recall += sum(cur_line)
            num_char += len(cur_line)
    print 'recall', recall / num_char


evaluate_mrr()
evaluate_recall()