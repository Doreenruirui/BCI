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

# def get_bin(prob_vec):
#     prob_vec = list(map(float, prob_vec.split('_')))
#     num_wit = len(prob_vec)
#     bin = [[] for _ in range(num_wit)]
#     bin[0].append(0)
#     bin_no = 0
#     for i in range(num_wit):
#         cur_prob = prob_vec[i]
#         if i == 0:
#             print(cur_prob - 0.5)
#             num_bin = int(np.ceil((cur_prob - 0.5) / 0.025))
#         else:
#             num_bin = int(cur_prob / 0.025)
#         print(num_bin)
#         bin[i] += [ele + bin_no + 1 for ele in range(num_bin)]
#
#         bin_no += num_bin
#     return bin
#
# num_group = np.zeros(5)
# total = 0
# nwrong = 0
# for sen, cand, prob in line_pairs:
#     for i in range(1, len(sen)):
#         if sen[i - 1] == cand[i][0]:
#             num_group[0] += 1
#         elif sen[i - 1] == cand[i][1]:
#             num_group[1] += 1
#         elif sen[i - 1] == cand[i][2]:
#             num_group[2] += 1
#         else:
#             num_group[3] += 1
#         if len(np.unique(cand[i])) < num_wit:
#             nwrong += 1
#     total += len(sen) - 1
# print(num_group, total, num_group / total, nwrong)

--dev="test" --size=512 --num_layers=3 --batch_size=128 --max_seq_len=300 --learning_rate=0.0003 --learning_rate_decay_factor=0.95 --optimizer="adam" --flag_generate=False --keep_prob=0.9 --data_dir="/home/rui/Dataset/BCI/NYT/1/1/1/0.0/sub" --train_dir="/home/rui/Model/BCI/NYT/1/1/1/lstm_fix_prior_3_1_prob_0.7_0.2" --prior_vec='3_1' --prob_vec='0.7_0.2' --num_wit=2
