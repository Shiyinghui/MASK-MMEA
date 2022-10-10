from __future__ import print_function
from __future__ import division
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time

import copy
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from config import Config
from tools import *


class DB15kImageDataset(Dataset):
    def __init__(self, idx_dir: str, img_dir: str, split: str, mode, input_size):
        super(DB15kImageDataset, self).__init__()
        with open(os.path.join(idx_dir, split+f'_{mode}.pkl'), 'rb') as f:
            info = pickle.load(f)
        self.split = split
        self.mode = mode  # item: (ent, type, img_name, label_id)
        self.images = [os.path.join(img_dir, item[2]+'.npy') for item in info]
        self.labels = [item[3] for item in info]
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        print(max(self.labels), min(self.labels), len(set(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.data_transforms[self.mode](Image.fromarray(np.load(self.images[index])))
        label = self.labels[index]
        return img, label


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def train_model(model, dataloaders, criterion, optimizer, device, lang,
                cfg, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                if epoch < int(num_epochs/2):
                    continue
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            iter_cnt = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if iter_cnt % 100 == 0:
                    print('{}: epoch {}, iter {}'.format(phase, epoch, iter_cnt))
                iter_cnt += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                #  forward， track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    #torch.save(model.state_dict(), f'./resnet_{lang}.pkl')
    save_model(model, cfg.model_path)
    return model, val_acc_history


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        # """ Resnet18
        # """
        # model_ft = models.resnet18(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

        """ Resnet152
                """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def main(cfg):
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    lang = cfg.ds  # 'fr_en' or 'ja_en' or 'zh_en'
    img_idx_dir = cfg.img_idx_dir
    img_data_dir = cfg.img_data_dir

    model_name = "resnet"
    num_classes = cfg.finetune_classes
    batch_size = 32
    num_epochs = 25   # 15
    feature_extract = False  # True

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    #print(model_ft)

    print("Initializing Datasets and Dataloaders...")
    train_dataset = DB15kImageDataset(img_idx_dir, img_data_dir, lang, 'train', input_size)
    val_dataset = DB15kImageDataset(img_idx_dir, img_data_dir, lang, 'val', input_size)

    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True,
                          collate_fn=dataset_collate)
    test_dl = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True,
                         collate_fn=dataset_collate)
    dataloaders_dict = {'train': train_dl, 'val': test_dl}

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                                 optimizer_ft, device, lang, cfg,
                                 num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))


def extract_feature(cfg, save_feat=False, save_pred=False):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # load data
    img_data_dir = cfg.img_data_dir
    val_ind_path = cfg.ill_img_data_path
    model_path = cfg.model_path
    save_feat_path = cfg.img_feature_path
    pred_path = cfg.pred_path
    pred_label = dict()

    with open(val_ind_path, 'rb') as f:
        data = pickle.load(f)
        # item: [ent, type, img_name, type_idx]
    images = [os.path.join(img_data_dir, item[2]+'.npy') for item in data]
    fn2ent = {item[2]: item[0] for item in data}
    img2label = {item[2]: item[3] for item in data}

    img_feature_dict = {}
    num_classes = cfg.finetune_classes
    model_ft = models.resnet152(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    state_dict = torch.load(model_path, map_location='cpu') #map_location={'cuda:0': 'cuda:1'})
    model_ft.load_state_dict(state_dict)

    layer = model_ft._modules.get('avgpool')  # Use the model object to select the desired layer
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluation mode

    scaler = transforms.Scale((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    batch_size = 8
    idx = 0
    topk = 5

    while idx < len(images):
        img_tensor = []
        print(idx)
        for img_file in images[idx:idx+batch_size]:
            pil_img = Image.fromarray(np.load(img_file))
            t_img = Variable(normalize(to_tensor(scaler(pil_img))))
            img_tensor.append(t_img)

        img_tensor = torch.stack(img_tensor)
        my_embedding = torch.zeros([img_tensor.shape[0], 2048, 1, 1])
        img_tensor = img_tensor.to(device)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = layer.register_forward_hook(copy_data)
        outputs = model_ft(img_tensor)
        #_, preds = torch.max(outputs, 1)
        _, preds = torch.sort(outputs, 1, descending=True)

        h.remove()
        my_embedding = torch.reshape(my_embedding, [img_tensor.shape[0], 2048, ]).numpy()
        for i, img_file in enumerate(images[idx:idx+batch_size]):
            img_name = img_file.rstrip('.npy').split('/')[-1]
            img_feature_dict[fn2ent[img_name]] = my_embedding[i]
            pred_label[img_name] = preds[i].cpu().numpy().tolist()[:topk]
        idx += 8
    print(len(set(pred_label).intersection(set(img2label))))

    # calculate accuracy
    cnt1, cnt2 = 0, 0
    for img, gt_label in img2label.items():
        if img in pred_label:
            if gt_label == pred_label[img][0]:
                cnt1 += 1
            if gt_label in pred_label[img]:
                cnt2 += 1

    print("h@1: {}, h@5:{}".format(cnt1/len(img2label), cnt2/len(img2label)))

    kg1_ent2id = get_ent2id(cfg.kg1_ent2id_path)
    kg2_ent2id = get_ent2id(cfg.kg2_ent2id_path)
    ent2id = dict(kg1_ent2id, **kg2_ent2id)
    img_feature_dict = {ent2id[k]: v for k, v in img_feature_dict.items()}

    if save_feat:
        with open(save_feat_path, 'wb') as f:
            pickle.dump(img_feature_dict, f)

    if save_pred:
        with open(pred_path, 'wb') as f:
            pickle.dump(pred_label, f)


if __name__ == "__main__":
    split = 'fr_en'  # 'fr_en' or 'ja_en' or 'zh_en'
    cfg = Config(split)
    #main(cfg)
    extract_feature(cfg, save_feat=True, save_pred=True)

