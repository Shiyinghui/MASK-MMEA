# @Time   : 2022/9/25 21:40
# @Author : syh

import pickle
import numpy as np
from config import Config
from tools import *
from analyze_clf import load_res, get_desc2base
from gen_mask import build_conflict_matrix, get_ft_info


def load_align_res(file_path):
    with open(file_path, 'r') as f:
        res = f.readlines()
    align_res = dict()
    for line in res[1:]:
        content = line.strip().split(',')[1:5]
        content = [int(item) for item in content]
        rank, query_id, gt_id, ret1 = content[:]
        align_res[query_id] = {'ret': ret1, 'ans': gt_id, 'rank': rank}
    return align_res


def get_align_error(file_path):
    with open(file_path, 'r') as f:
        res = f.readlines()
    error_info = dict()
    q_ids = set()
    for line in res[1:]:
        content = line.strip().split(',')[1:5]
        content = [int(item) for item in content]
        rank, query_id, gt_id, ret1 = content[:]
        q_ids.add(query_id)
        if rank > 0:
            error_info[query_id] = {'ret': ret1, 'ans': gt_id, 'rank': rank}
    return error_info


def cal_sim(id2feat, id1, id2):
    cos_sim = np.dot(id2feat[id1], id2feat[id2]) / \
              (np.linalg.norm(id2feat[id1]) * np.linalg.norm(id2feat[id2]))
    return cos_sim


def has_img(gt, pred, ent2type):
    t1, t2 = 0, 0
    if gt in ent2type:
        t1 = 1
    if pred in ent2type:
        t2 = 1
    return t1, t2


def merged_ent2degree(cfg: Config):
    e2d1: dict = get_ent2degree(cfg.kg1_triple_path)
    e2d2: dict = get_ent2degree(cfg.kg2_triple_path)
    e2d = dict()
    ills = get_ills(cfg.ills_path)
    for (id1, id2) in ills:
        e2d[(id1, id2)] = e2d1[id1] + e2d2[id2]
    return e2d


def ana(data: list, ent2clf, id2ent: dict, cft, ent2mask):
    ranks = []
    cnt = 0

    sim_1, sim_2, sim_diff = [], [], []
    #id2feat = pickle.load(open(cfg.img_feature_path, 'rb'))
    by_type = defaultdict(int)
    by2 = defaultdict(list)

    keys = {v[0] for v in ent2clf.values()}
    desc2base = get_desc2base(keys)
    e2d = merged_ent2degree(cfg)
    degs = []

    for item in data:
        src, gt, pred, rank = item
        qid, gt_id, pred_id = id2ent[src], id2ent[gt], id2ent[pred]
        degs.append(e2d[(qid, gt_id)])

        if src in ent2mask and ent2mask[src] == 1 \
            and gt in ent2mask and ent2mask[gt] == 1 \
                and pred not in ent2mask:
            cnt += 1
        if src in ent2clf and gt in ent2clf and pred in ent2clf:
            p1, p2, p3 = ent2clf[src][1], ent2clf[gt][1], ent2clf[pred][1]
            t1, t2, t3 = ent2clf[src][0], ent2clf[gt][0], ent2clf[pred][0]

            # if ent2mask[src] == 1 and ent2mask[gt] == 0 and ent2mask[pred] == 0:
            #     cnt += 1
            #     by2[(desc2base[t1], desc2base[t2], desc2base[t3])].append((src, gt, pred))
            #     print(t1, p1, t2, p2, t3, p3)
            #     sim1 = cal_sim(id2feat, qid, gt_id)
            #     sim2 = cal_sim(id2feat, qid, pred_id)
            #     print(sim1, sim2)
            #     sim_1.append(sim1)
            #     sim_2.append(sim2)
            #     sim_diff.append(sim1-sim2)
            #     ranks.append(rank)

    # ranks = np.array(ranks)
    # print(len(ranks), len(ranks[ranks >= 10]))
    # print(sum(sim_diff)/len(sim_diff))
    # print(sum(sim_1)/len(sim_1), sum(sim_2)/len(sim_2))
    # print(cnt)
    # for k, v in by_type.items():
    #     print(k, v)
    #
    # cnt = 0
    # by3 = defaultdict(int)
    # for k, v in by2.items():
    #     if k[0] == k[-1]:
    #         cnt += len(v)
    #         by3[k[0]] += len(v)
    #         continue
    #     print(k, len(v))
    #     for i in v:
    #         print(i)
    # print(cnt)
    # for k, v in by3.items():
    #     print(k, v)

    degs = np.array(degs)
    avg_deg = sum(e2d.values()) / len(e2d)
    print(avg_deg, np.mean(degs), len(degs[degs <= avg_deg]))


def collect(query_ids, data: dict, id2ent: dict):
    res = []
    for qid in query_ids:
        gt_id, pred_id = data[qid]['ans'], data[qid]['ret']
        src, gt, pred = id2ent[qid], id2ent[gt_id], id2ent[pred_id]
        rank = data[qid]['rank']
        res.append((src, gt, pred, rank))
    return res


# def analyze(cfg: Config):
#     struct_res: dict = load_align_res(cfg.struct_align_pred_path)
#     struct_err = {k: v for k, v in struct_res.items() if v['ret'] != v['ans']}
#     mm_res: dict = load_align_res(cfg.cft_align_pred_path)
#     mm_err = {k: v for k, v in mm_res.items() if v['ret'] != v['ans']}
#
#     ea_res_path = '/code/syh/eva/mask_ea/logs/fr_en/pred_eva.txt'
#     ea_res: dict = load_align_res(ea_res_path)
#     common1 = set(struct_res) & set(ea_res)
#     common2 = set(struct_res) & set(mm_res)
#
#     w2r_qid = set(struct_err) - set(mm_err)
#     r2w_qid = set(mm_err) - set(struct_err)
#
#     ent2clf = load_res(cfg)
#     ent2id, id2ent = get_ent_id(cfg)
#     # load ent mask
#     with open(cfg.mask_path, 'rb') as f:
#         ent2mask: dict = pickle.load(f)
#
#     # get the conflicting dict
#     cft: dict = build_conflict_matrix(cfg)
#     _, type2idx = get_ft_info(cfg)
#     idx2type = {v: k for k, v in type2idx.items()}
#     cft = {(idx2type[k[0]], idx2type[k[1]]): v for k, v in cft.items()}
#
#     w2r_data: list = collect(w2r_qid, struct_res, id2ent)
#     r2w_data: list = collect(r2w_qid, mm_res, id2ent)
#     ana(w2r_data, ent2clf, ent2id, cft, ent2mask)
#     #ana(r2w_data, ent2clf, ent2id, cft, ent2mask)


# def analyze(cfg: Config):
#     struct_res: dict = load_align_res(cfg.struct_align_pred_path)
#     struct_err = {k: v for k, v in struct_res.items() if v['ret'] != v['ans']}
#     mm_res: dict = load_align_res(cfg.cft_align_pred_path)
#     mm_err = {k: v for k, v in mm_res.items() if v['ret'] != v['ans']}
#
#     ea_res_path = '/code/syh/eva/mask_ea/logs/fr_en/pred_eva.txt'
#     ea_res: dict = load_align_res(ea_res_path)
#     common1 = set(struct_res) & set(ea_res)
#     common2 = set(struct_res) & set(mm_res)
#
#     w2r_qid = set(struct_err) - set(mm_err)
#     r2w_qid = set(mm_err) - set(struct_err)
#
#     ent2clf = load_res(cfg)
#     ent2id, id2ent = get_ent_id(cfg)
#     # load ent mask
#     with open(cfg.mask_path, 'rb') as f:
#         ent2mask: dict = pickle.load(f)
#
#     # get the conflicting dict
#     cft: dict = build_conflict_matrix(cfg)
#     _, type2idx = get_ft_info(cfg)
#     idx2type = {v: k for k, v in type2idx.items()}
#     cft = {(idx2type[k[0]], idx2type[k[1]]): v for k, v in cft.items()}
#
#     w2r_data: list = collect(w2r_qid, struct_res, id2ent)
#     r2w_data: list = collect(r2w_qid, mm_res, id2ent)
#     ana(w2r_data, ent2clf, ent2id, cft, ent2mask)
#     #ana(r2w_data, ent2clf, ent2id, cft, ent2mask)


def load_mask(fp):
    with open(fp, 'rb') as f:
        mask = pickle.load(f)
    return mask


def analyze_error(cfg):
    # query_id: {'ret': ret1, 'ans': gt_id, 'rank': rank}
    struct_err: dict = get_align_error(cfg.res_struct_path)
    #mask_err: dict = get_align_error(cfg.res_mask_path)
    sv_err: dict = get_align_error(cfg.res_sv_path)
    #post_err: dict = load_align_res(cfg.res_spec_path)

    # ent2mask: dict = load_mask(cfg.mask_path)
    # ent2svmask: dict = load_mask(cfg.sv_mask_path)
    # ent2id, id2ent = get_ent_id(cfg)

    #c1 = set(struct_err) & set(mask_err)
    c2 = set(struct_err) & set(sv_err)
    #c3 = set(struct_err) & set(post_err)

    #r2w1 = set(mask_err) - set(struct_err)
    r2w2 = set(sv_err) - set(struct_err)
    #r2w3 = set(post_err) - set(struct_err)

    #w2r1 = set(struct_err) - set(mask_err)
    w2r2 = set(struct_err) - set(sv_err)
    #w2r3 = set(struct_err) - set(post_err)

    # print('struct:', len(struct_err), 'mask:', len(mask_err), 's+v:', len(sv_err), 'post:', len(post_err))
    # print('common:', 'mask:', len(c1), 's+v:', len(c2), 'post:', len(c3))
    # print('r2w:', 'mask:', len(r2w1), 's+v:', len(r2w2), 'post:', len(r2w3))
    # print('w2r:', 'mask:', len(w2r1), 's+v:', len(w2r2), 'post:', len(w2r3))
    refine_mask(cfg, struct_err, sv_err, cfg.sv_mask_path, cfg.post_mask_path)


def refine_mask(cfg: Config, a1_err: dict, a2_err: dict, ori_mask_path, new_mask_path):
    """
    :param cfg: config
    :param a1_err: info of alignments errors using approach 1
    :param a2_err: info of alignments errors using approach 2
    :param ori_mask_path: original mask path
    :param new_mask_path: the path to save the new_mask
    """
    new = set(a2_err) - set(a1_err)
    ent2mask: dict = pickle.load(open(ori_mask_path, 'rb'))
    kg1_ent2id: dict = get_ent2id(cfg.kg1_ent2id_path)
    kg2_ent2id: dict = get_ent2id(cfg.kg2_ent2id_path)
    ent2id = dict(kg1_ent2id, **kg2_ent2id)
    id2ent = {v: k for k, v in ent2id.items()}

    cnt = 0
    for query_id in new:
        ent = id2ent[query_id]
        if ent in ent2mask and ent2mask[ent] > 0:
            cnt += 1
            ent2mask[ent] = 0
    print(cnt)
    with open(new_mask_path, 'wb') as f:
        pickle.dump(ent2mask, f)


if __name__ == "__main__":
    split = 'zh_en'
    cfg = Config(split)
    analyze_error(cfg)
    #merged_ent2degree(cfg)


