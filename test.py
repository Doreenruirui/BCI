# mrr_dist = [[] for _ in range(11)]
# for mrr, err in zip(new_mrr, err_id):
#     tmp = [[] for _ in range(11)]
#     for i in range(1, len(mrr)):
#         num = 0
#         for j in range(1, 11):
#             ind = i - j
#             dist = j
#             if ind >= 0:
#                 if ind in err:
#                     num += 1
#             tmp[dist].append(num)
#     for i in range(11):
#         mrr_dist[i].append(tmp[i])
#
# dist = 3
# def compute_err(dist, new_mrr):
#     err_dist = [[] for _ in range(dist + 1)]
#     for mrr, err in zip(new_mrr, mrr_dist[dist]):
#         for j in range(1, len(mrr)):
#             err_dist[err[j - 1]].append(mrr[j])
#     return err_dist
#
# def compute_pos(dist, mrr_dist, new_mrr):
#     err_dist = [[] for _ in range(dist)]
#     for i in range(len(mrr_dist[1])):
#         for j in range(1, len(new_mrr[i])):
#             if mrr_dist[dist][i][j - 1] == 1:
#                 for k in range(dist - 1):
#                     if mrr_dist[dist - k][i][j - 1] == 1 and mrr_dist[dist- k - 1][i][j - 1] == 0:
#                         err_dist[dist - k - 1].append(new_mrr[i][j])
#                 if mrr_dist[1][i][j - 1] == 1:
#                     err_dist[0].append(new_mrr[i][j])
#     return err_dist
#
#
# for line in f_in:
#     line = line.strip().lower()
#     items = list(line)
#     len_line = len(line)
#     index = np.arange(len_line)
#     np.random.shuffle(index)
#     nchoice = int(np.ceil(len_line * 0.2))
#     chose_id = index[:nchoice]
#     for cid in chose_id:
#         cur_char = line[cid]
#         cands = [' '] + list(string.ascii_lowercase)
#         cands.remove(cur_char)
#         cand_id = np.random.choice(26, 1)
#         rep = cands[cand_id[0]]
#         items[cid] = rep
#     f_out.write(''.join(items) + '\n')

