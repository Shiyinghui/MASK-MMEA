# @Time   : 2022/9/25 21:40
# @Author : syh
import pickle
from config import Config
from collections import defaultdict
from process_type import get_sub2super, get_sibling, get_type2path, get_type2desc


# load prediction results
def load_res(cfg):
    pred_data_path = cfg.pred_path
    val_data_path = cfg.ill_img_data_path
    with open(val_data_path, 'rb') as f:
        val_data: list = pickle.load(f)  # [ent, label, img_name, label_idx]
    with open(pred_data_path, 'rb') as f:
        pred_data: dict = pickle.load(f)  # img_name: [pred_1, pred_2, pred_3, pred_4, pred_5]

    gt = {item[0]: item[1] for item in val_data}
    imgfn2ent = {item[2]: item[0] for item in val_data}
    id2type = {item[3]: item[1] for item in val_data}
    pred = dict()
    for img in pred_data:
        ent = imgfn2ent[img]
        pred[ent] = [id2type[i] for i in pred_data[img]]
    assert len(pred) == len(gt)

    ent2clf = dict()
    for ent in pred:
        gt_type, pred_type = gt[ent], pred[ent][0]
        if pred_type == gt_type:
            ent2clf[ent] = (gt_type, pred_type, 1)
        else:
            ent2clf[ent] = (gt_type, pred_type, 0)
    return ent2clf


# ent2clf, key: ent, value: (gt_type, pred_type, 0 or 1)
def get_type2acc(ent2clf: dict):
    type2pred = defaultdict(list)
    for v in ent2clf.values():
        type2pred[v[0]].append(v[-1])
    type2acc = {k: (sum(v)/len(v), len(v)) for k, v in type2pred.items()}
    # type2acc, key: type, value: (acc, number of entities of this type)
    type2acc = sorted(type2acc.items(), key=lambda x: x[1][0], reverse=False)
    return type2acc


# print accuracy per class
def print_acc(type2acc):
    sub2super = get_sub2super()
    type2path = get_type2path(sub2super)
    res = dict()
    for (t, acc) in type2acc:
        output = ''
        for elem in reversed(type2path[t][:-1]):
            output += elem + ' '
        res[output.rstrip()] = acc
    res = sorted(res.items(), key=lambda x: x[0])
    for (k, v) in res:
        print(k, v)
    return res


# ent2clf, key: ent, value: (gt_type, pred_type, 0 or 1)
def clf_errors(ent2clf: dict):
    siblings = get_sibling()
    sub2super = get_sub2super()
    type2path = get_type2path(sub2super)
    errors = {k: v for k, v in ent2clf.items() if v[-1] == 0}
    mis_l1, mis_l2, mis_l3 = defaultdict(list), defaultdict(list), defaultdict(list)
    for k, v in errors.items():
        gt, pred = v[:2]
        if pred in type2path[gt] or gt in type2path[pred]:
            mis_l1[(gt, pred)].append(k)
        elif pred in siblings[gt]:
            mis_l2[(gt, pred)].append(k)
        else:
            mis_l3[(gt, pred)].append(k)
    return mis_l1, mis_l2, mis_l3


def cross_domain_error(mis_pred: dict):
    wrong_k = {k[1] for k in mis_pred}
    desc2base = get_desc2base(wrong_k)
    to_type = defaultdict(int)
    err_cnt = 0
    for k, v in mis_pred.items():
        to_type[desc2base[k[1]]] += len(v)
        err_cnt += len(v)
    to_type = sorted(to_type.items(), key=lambda x: x[1], reverse=True)
    for k, v in to_type:
        print(k, v, v/err_cnt)


# get prediction results of a specific class and all its descendant classes
def pred_by_domain(ent2clf: dict, spec_type):
    type2desc: dict = get_type2desc()
    types = {spec_type}
    if spec_type in type2desc:
        types |= type2desc[spec_type]
    res_ent2clf = {k: v for k, v in ent2clf.items() if v[0] in types}
    return res_ent2clf


def mis_pred(ent2clf: dict):
    errors = {k: v for k, v in ent2clf.items() if v[-1] == 0}
    s2t = defaultdict(list)
    for k, v in errors.items():
        s2t[v].append(k)
    s2t = sorted(s2t.items(), key=lambda x: len(x[1]), reverse=True)
    for (k, v) in s2t:
        print(k, len(v))
    return s2t


def get_desc2base(keys: set):
    desc2base = dict()
    sub2super = get_sub2super()
    type2path = get_type2path(sub2super)
    for k in keys:
        if k in type2path:
            desc2base[k] = type2path[k][-2]
            if type2path[k][-2] == 'Agent' and len(type2path[k]) >= 3:
                desc2base[k] = type2path[k][-3]
    return desc2base


def analyze():
    cfg = Config('fr_en')
    ent2clf = load_res(cfg)
    classes = set([v[0] for v in ent2clf.values()])
    for split in ['zh_en', 'ja_en']:
        cfg = Config(split)
        cur_res: dict = load_res(cfg)
        for ent in cur_res:
            # if ent not in ent2clf and cur_res[ent][0] in classes:
            #     ent2clf[ent] = cur_res[ent]
            if ent not in ent2clf:
                ent2clf[ent] = cur_res[ent]

    ent2clf = pred_by_domain(ent2clf, 'TelevisionSeason')
    mis_l1, mis_l2, mis_l3 = clf_errors(ent2clf)
    #cross_domain_error(mis_l3)

    # for k, v in ent2clf.items():
    #     if v[-1] == 1:
    #         print(k)

    # cnt1 = sum([len(v) for v in mis_l1.values()])
    # cnt2 = sum([len(v) for v in mis_l2.values()])
    # cnt3 = sum([len(v) for v in mis_l3.values()])
    # acc = sum([v[-1] for v in ent2clf.values()]) / len(ent2clf)
    # print(acc, (len(ent2clf)-cnt3)/len(ent2clf))
    # print(cnt1/len(ent2clf), cnt2/len(ent2clf), cnt3/len(ent2clf))
    mis_l3 = sorted(mis_l3.items(), key=lambda x: len(x[1]), reverse=True)
    for k, v in mis_l3:
        print(k, len(v))
        for elem in v:
            print(elem)

    # mis_l3 = sorted(mis_l3.items(), key=lambda x: len(x[1]), reverse=True)
    # for k, v in mis_l3:
    #     print(k, len(v))

    # type2acc = get_type2acc(ent2clf)
    # print_acc(type2acc)

    # type2acc = print_acc(type2acc)
    # can = {'Work', 'Organisation', 'Person', 'Place'}
    # cnt1, cnt2 = 0, 0
    # for (k, v) in type2acc:
    #     cs = k.split()
    #     for i in can:
    #         if i in cs:
    #             cnt1 += v[1]
    #     cnt2 += v[1]
    # print(cnt1/cnt2)

    # acc = sum([v[-1] for v in ent2clf.values()]) / len(ent2clf)
    # print(acc)
    # with open('./type2acc.pkl', 'wb') as f:
    #     pickle.dump(res, f)


if __name__ == "__main__":
    analyze()
