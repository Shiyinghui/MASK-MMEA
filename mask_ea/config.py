split = "zh_en"  # [fr_en, ja_en, zh_en]
data_dir = f"../data/DBP15K/{split}"
img_feature_path = f"../img_data/retrain/{split}/{split}_feature.pkl"
#img_feature_path = f"/code/syh/mmea-imp/data/visual/DBP15K/{split}_res152_feature.pkl"
use_mask = True
if use_mask:
    mask_path = f"../img_data/retrain/{split}/{split}_post_mask.pkl"
else:
    mask_path = None

res_path = f"logs/{split}/pred_nca_spec.txt"
ent_type_dir = f"../data/DBP15K_type/{split}"
train_rate = 0.3  # training set rate
cuda_num = 0
seed = 2021
epochs = 1000  # number of epochs to train
batch_size = 7500
check_point = 50
img_emb_dim = 200
gcn_hidden_units = [400, 400, 200]
# img_emb_dim = 400
# gcn_hidden_units = [400, 400, 400]
lr = 0.0005  # initial learning rate
instance_normalization = False  # enable instance normalization
weight_day = 0  # weight decay (L2 loss on parameters)
dropout = 0.0  # dropout rate for layers
dist = 2   # L1 distance or L2 distance
csls = True  # use CSLS for inference
csls_k = 3  # top k for csls



