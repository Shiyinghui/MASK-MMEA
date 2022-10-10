import os
import pickle


class Config:
    def __init__(self, split='fr_en'):
        ds = split  # 'fr_en', 'ja_en', 'zh_en'
        project_dir = '/code/syh/MASK-MMEA/'
        data_dir = project_dir + f'/data/DBP15K/{ds}'

        self.ds = split
        self.base_dir = project_dir
        self.kg1_ent2id_path = data_dir + '/ent_ids_1'
        self.kg2_ent2id_path = data_dir + '/ent_ids_2'
        self.ills_path = data_dir + '/ill_ent_ids'
        self.kg1_triple_path = data_dir + '/triples_1'
        self.kg2_triple_path = data_dir + '/triples_2'

        self.ent_type_dir = project_dir + '/data/DBP15K_type'
        self.ontology_file_path = self.ent_type_dir + '/onto_subClassOf_triples'
        self.disjoint_info = project_dir + '/data/DBP15K_type/disjoint.txt'

        self.img_idx_dir = project_dir + f'/img_data/retrain/{ds}'
        #self.img_data_dir = project_dir + '/img_data/all_images/'
        self.img_data_dir = '/code/syh/eva/raw_data/np_data/'
        self.ent2imgfn_path = project_dir + '/img_data/ent2imgfn.pkl'

        self.model_path = project_dir + f'/img_data/retrain/{ds}/{ds}_model.pth'
        self.train_img_data_path = project_dir + f'/img_data/retrain/{ds}/{ds}_train.pkl'
        # ill: inter-language links
        self.ill_img_data_path = project_dir + f'/img_data/retrain/{ds}/{ds}_test.pkl'

        self.img_feature_path = project_dir + f'/img_data/retrain/{ds}/{ds}_feature.pkl'
        self.mask_path = project_dir + f'/img_data/retrain/{ds}/{ds}_mask.pkl'
        self.pred_path = project_dir + f'/img_data/retrain/{ds}/{ds}_pred.pkl'
        self.id2type_path = project_dir + f'/img_data/retrain/{ds}/{ds}_id2type.pkl'

        self.sv_mask_path = project_dir + f'/img_data/retrain/{ds}/{ds}_mask_1.0.pkl'
        self.post_mask_path = project_dir + f'/img_data/retrain/{ds}/{ds}_post_mask.pkl'

        # self.res_struct_path = project_dir + f'mask_ea/logs/{ds}/pred_struct.txt'
        # self.res_avg_path = project_dir + f'mask_ea/logs/{ds}/pred_avg.txt'
        # self.res_cat_path = project_dir + f'mask_ea/logs/{ds}/pred_cat.txt'
        # self.res_mask_path = project_dir + f'mask_ea/logs/{ds}/pred_mask0.txt'
        # self.res_sv_path = project_dir + f'mask_ea/logs/{ds}/pred_mask1.txt'
        # self.res_spec_path = project_dir + f'mask_ea/logs/{ds}/pred_spec.txt'

        self.res_struct_path = project_dir + f'mask_ea/logs/{ds}/pred_struct_200.txt'
        self.res_avg_path = project_dir + f'mask_ea/logs/{ds}/pred_avg.txt'
        self.res_cat_path = project_dir + f'mask_ea/logs/{ds}/pred_cat.txt'
        self.res_mask_path = project_dir + f'mask_ea/logs/{ds}/pred_mask0.txt'
        self.res_sv_path = project_dir + f'mask_ea/logs/{ds}/pred_nca_mask1.txt'
        self.res_spec_path = project_dir + f'mask_ea/logs/{ds}/pred_spec.txt'

        if os.path.exists(self.ill_img_data_path):
            data = pickle.load(open(self.ill_img_data_path, 'rb'))
            self.finetune_classes = len({item[3] for item in data})
