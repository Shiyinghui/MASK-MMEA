# @Time   : 2022/9/21 15:12
# @Author : syh
from utils import *
from model import *
import gc
import config
import torch.optim as optim
import os
import time


def load_data():
    lang_list = [1, 2]
    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(config.data_dir, lang_list)
    e1 = os.path.join(config.data_dir, 'ent_ids_1')
    e2 = os.path.join(config.data_dir, 'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)
    adj = get_adjr(ENT_NUM, triples, norm=True)
    img_features, _ = load_img(ENT_NUM, config.img_feature_path)
    return ent2id_dict, ENT_NUM, REL_NUM, left_ents, right_ents, ills, adj, img_features


def set_seed():
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    # set seed
    set_seed()
    # use gpu
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # load data
    ent2id_dict, ENT_NUM, REL_NUM, \
    left_ents, right_ents, ills, adj, img_features = load_data()

    np.random.shuffle(ills)
    train_ill = np.array(ills[:int(len(ills) // 1 * config.train_rate)], dtype=np.int32)
    test_ill = np.array(ills[int(len(ills) // 1 * config.train_rate):], dtype=np.int32)
    test_left = torch.LongTensor(test_ill[:, 0].squeeze()).to(device)
    test_right = torch.LongTensor(test_ill[:, 1].squeeze()).to(device)

    # prepare model, move data to gpu
    input_ent_idx = torch.LongTensor(np.arange(ENT_NUM)).to(device)
    entity_emb = nn.Embedding(ENT_NUM, config.gcn_hidden_units[0])
    nn.init.normal_(entity_emb.weight, std=1.0 / math.sqrt(ENT_NUM))
    entity_emb.requires_grad = True
    entity_emb = entity_emb.to(device)

    adj = adj.to(device)
    img_features = F.normalize(torch.Tensor(img_features).to(device))
    img_fc = nn.Linear(img_features.shape[1], config.img_emb_dim).to(device)

    gcn_units = config.gcn_hidden_units
    cross_graph_model = GCN(gcn_units[0], gcn_units[1], gcn_units[2], dropout=config.dropout).to(device)

    for param in img_fc.parameters():
        param.requires_grad = True

    params = [
        {"params":
             list(cross_graph_model.parameters()) +
             list(img_fc.parameters()) +
             [entity_emb.weight]
         }]
    optimizer = torch.optim.AdamW(
        params,
        lr=config.lr
    )

    print("GCN model details:")
    print(cross_graph_model)
    print("optimiser details:")
    print(optimizer)

    # modality-specific loss
    criterion_gcn = NCA_loss(alpha=5, beta=10, ep=0.0)
    criterion_img = NCA_loss(alpha=15, beta=10, ep=0.0)

    # wae, weighted average embedding
    criterion_wae = NCA_loss(alpha=15, beta=10, ep=0.0)

    # Train
    print("[start training...] ")
    t_total = time.time()

    acc_across_epochs = []

    for epoch in range(config.epochs):
        t_epoch = time.time()
        cross_graph_model.train()
        img_fc.train()
        optimizer.zero_grad()

        struct_emb = cross_graph_model(entity_emb(input_ent_idx), adj)
        visual_emb = img_fc(img_features)

        s_emb = F.normalize(struct_emb.detach())
        v_emb = F.normalize(visual_emb.detach())

        avg_emb = s_emb + v_emb
        lib_cosine_sim = torch.nn.CosineSimilarity(dim=1)
        cos_sim_mat_struct = lib_cosine_sim(s_emb, avg_emb)
        cos_sim_mat_visual = lib_cosine_sim(v_emb, avg_emb)
        cos_sim_mat = cos_sim_mat_struct + cos_sim_mat_visual
        struct_weight = torch.unsqueeze(torch.div(cos_sim_mat_struct, cos_sim_mat), dim=1)
        visual_weight = torch.unsqueeze(torch.div(cos_sim_mat_visual, cos_sim_mat), dim=1)
        weighted_avg_emb = struct_weight * s_emb + visual_weight * v_emb

        loss_sum_all = 0
        # manual batching
        np.random.shuffle(train_ill)
        for si in np.arange(0, train_ill.shape[0], config.batch_size):
            loss_GCN = criterion_gcn(struct_emb, train_ill[si:si+config.batch_size], [], device=device)
            loss_img = criterion_img(visual_emb, train_ill[si:si+config.batch_size], [], device=device)
            loss_wae = criterion_wae(weighted_avg_emb, train_ill[si:si+config.batch_size], [], device=device)
            loss_all = loss_GCN + loss_img + loss_wae
            loss_all.backward(retain_graph=True)
            loss_sum_all = loss_sum_all + loss_all.item()
        optimizer.step()
        print("[epoch {:d}] loss_all: {:f}, time: {:.4f} s".format(epoch, loss_sum_all, time.time() - t_epoch))

        del struct_emb, visual_emb, weighted_avg_emb

        # Test
        if (epoch+1) % config.check_point == 0:
            print("\n[epoch {:d}] checkpoint!".format(epoch))

            with torch.no_grad():
                cross_graph_model.eval()
                img_fc.eval()

                struct_emb = cross_graph_model(entity_emb(input_ent_idx), adj)
                visual_emb = img_fc(img_features)

                s_emb = F.normalize(struct_emb.detach())
                v_emb = F.normalize(visual_emb.detach())

                avg_emb = s_emb + v_emb
                lib_cosine_sim = torch.nn.CosineSimilarity(dim=1)
                cos_sim_mat_struct = lib_cosine_sim(s_emb, avg_emb)
                cos_sim_mat_visual = lib_cosine_sim(v_emb, avg_emb)
                cos_sim_mat = cos_sim_mat_struct + cos_sim_mat_visual
                struct_weight = torch.unsqueeze(torch.div(cos_sim_mat_struct, cos_sim_mat), dim=1)
                visual_weight = torch.unsqueeze(torch.div(cos_sim_mat_visual, cos_sim_mat), dim=1)
                weighted_avg_emb = struct_weight * s_emb + visual_weight * v_emb

                weighted_avg_emb = F.normalize(weighted_avg_emb)
                sim = weighted_avg_emb[test_left].mm(weighted_avg_emb[test_right].t())
                final_sim = csls_sim(sim, config.csls_k)

                top_k = [1, 10]
                acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
                acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
                test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.

                if epoch + 1 == config.epochs:
                    to_write = []
                    test_left_np = test_left.cpu().numpy()
                    test_right_np = test_right.cpu().numpy()
                    to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])

                for idx in range(test_left.shape[0]):
                    values, indices = torch.sort(final_sim[idx, :], descending=True)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_l2r += (rank + 1)
                    mrr_l2r += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_l2r[i] += 1
                    if epoch + 1 == config.epochs:
                        indices = indices.cpu().numpy()
                        to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]],
                                         test_right_np[indices[1]], test_right_np[indices[2]]])

                if epoch + 1 == config.epochs:
                    import csv
                    with open(config.res_path, "w") as f:
                        wr = csv.writer(f, dialect='excel')
                        wr.writerows(to_write)

                mean_l2r /= test_left.size(0)
                mrr_l2r /= test_left.size(0)
                for i in range(len(top_k)):
                    acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
                    acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)

                print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}".format(top_k, acc_l2r, mean_l2r, mrr_l2r))
                acc_across_epochs.append(acc_l2r[0])
                del final_sim, struct_emb, visual_emb
                gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(acc_across_epochs)
    print("[optimization finished!]")
    print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))


if __name__ == "__main__":
    main()


