import random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from domainbed.lib import misc
from domainbed.match import algorithms, hparams_registry
from domainbed import datasets
import matplotlib.pyplot as plt
import copy
import numpy as np


class IndexWrapper:
    def __init__(self, dataset, use_raw_index=False):
        super().__init__()
        self.dataset = dataset
        if use_raw_index and isinstance(self.dataset, misc._SplitDataset):
            self.idx_map = self.dataset.keys
        else:
            self.idx_map = None
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.idx_map is not None:
            index = self.idx_map[index]
        return index, x, y

    def __len__(self):
        return len(self.dataset)

### matching sampling strategy ###

def dist_mat(x1, x2, func="l2", device='cuda'):
    '''
    compute distance between any two rows from two matrices
    try to parallel compute as much as possible
    '''
    flatten_x1 = x1.view(len(x1), -1).to(device)
    flatten_x2 = x2.view(len(x2), -1).to(device)
    if func == 'l1':
        try:
            return torch.cdist(flatten_x1, flatten_x2, p=1)
        except:
            try:
                result = []
                for i, x1_i in enumerate(flatten_x1):
                    sim = torch.abs(x1_i.unsqueeze(0) - flatten_x2).sum(-1)
                    result.append(sim)
                return torch.stack(result, 0)
            except:
                try:
                    result = []
                    for i, x2_i in enumerate(flatten_x2):
                        sim = torch.abs(x2_i.unsqueeze(0) - flatten_x1).sum(-1)
                        result.append(sim)
                    return torch.stack(result, 1)
                except:
                    result = torch.zeros([len(x1), len(x2)], device=device)
                    for i, x1_i in enumerate(flatten_x1):
                        for j, x2_j in enumerate(flatten_x2):
                            sim = torch.abs(x1_i - x2_j).sum()
                            result[i][j] = sim
                    return result
    elif func == 'l2':
        try:
            return torch.cdist(flatten_x1, flatten_x2, p=2)
        except:
            try:
                result = []
                for i, x1_i in enumerate(flatten_x1):
                    sim = (x1_i.unsqueeze(0) - flatten_x2).pow(2).sum(-1).sqrt()
                    result.append(sim)
                return torch.stack(result, 0)
            except:
                try:
                    result = []
                    for i, x2_i in enumerate(flatten_x2):
                        sim = (x2_i.unsqueeze(0) - flatten_x1).pow(2).sum(-1).sqrt()
                        result.append(sim)
                    return torch.stack(result, 1)
                except:
                    result = torch.zeros([len(x1), len(x2)], device=device)
                    for i, x1_i in enumerate(flatten_x1):
                        for j, x2_j in enumerate(flatten_x2):
                            sim = (x1_i - x2_j).pow(2).sum().sqrt()
                            result[i][j] = sim
                    return result
    elif func == 'linf':
        try:
            return torch.cdist(flatten_x1, flatten_x2, p=float('inf'))
        except:
            try:
                result = []
                for i, x1_i in enumerate(flatten_x1):
                    sim = torch.max(torch.abs(x1_i.unsqueeze(0) - flatten_x2), dim=-1)
                    result.append(sim)
                return torch.stack(result, 0)
            except:
                try:
                    result = []
                    for i, x2_i in enumerate(flatten_x2):
                        sim = torch.max(torch.abs(x2_i.unsqueeze(0) - flatten_x1), dim=-1)
                        result.append(sim)
                    return torch.stack(result, 1)
                except:
                    result = torch.zeros([len(x1), len(x2)], device=device)
                    for i, x1_i in enumerate(flatten_x1):
                        for j, x2_j in enumerate(flatten_x2):
                            sim = torch.max(torch.abs(x1_i - x2_j))
                            result[i][j] = sim
                    return result
    elif func[0]=='l' and len(func)==2:
        p = int(func[1])
        try:
            return torch.cdist(flatten_x1, flatten_x2, p=p)
        except:
            try:
                result = []
                for i, x1_i in enumerate(flatten_x1):
                    diff = x1_i.unsqueeze(0) - flatten_x2
                    sim = torch.sign(diff) * torch.abs(diff).pow(p).sum(-1).pow(1/p)
                    result.append(sim)
                return torch.stack(result, 0)
            except:
                try:
                    result = []
                    for i, x2_i in enumerate(flatten_x2):
                        diff = x2_i.unsqueeze(0) - flatten_x1
                        sim = torch.sign(diff) * torch.abs(diff).pow(p).sum(-1).pow(1/p)
                        result.append(sim)
                    return torch.stack(result, 1)
                except:
                    result = torch.zeros([len(x1), len(x2)], device=device)
                    for i, x1_i in enumerate(flatten_x1):
                        for j, x2_j in enumerate(flatten_x2):
                            diff = x1_i - x2_j
                            sim = torch.sign(diff) * torch.abs(diff).pow(p).sum().pow(1/p)
                            result[i][j] = sim
                    return result
    elif func == 'kld': 
        try:
            temp1 = flatten_x1.unsqueeze(1).repeat(1, len(x2), 1)
            temp2 = torch.transpose(flatten_x2.unsqueeze(1).repeat(1, len(x1), 1), 0, 1)
            return torch.sum(temp1*torch.log(temp1/temp2), -1)
        except:
            try:
                result = []
                for i, x1_i in enumerate(flatten_x1):
                    x1_i = x1_i.unsqueeze(0)
                    sim = torch.sum(x1_i*torch.log(x1_i/flatten_x2), -1)
                    result.append(sim)
                return torch.stack(result, 0)
            except:
                try:
                    result = []
                    for i, x2_i in enumerate(flatten_x2):
                        x2_i = x2_i.unsqueeze(0)
                        sim = torch.sum(flatten_x1*torch.log(flatten_x1/x2_i), -1)
                        result.append(sim)
                    return torch.stack(result, 1)
                except:
                    result = torch.zeros([len(x1), len(x2)], device=device)
                    for i, x1_i in enumerate(flatten_x1):
                        for j, x2_j in enumerate(flatten_x2):
                            sim = torch.sum(x1_i*torch.log(x1_i/x2_j))
                            result[i][j] = sim
                    return result

    else:
        print(func + " distance is not implemented.")
        exit(1)
    
class CFmatchWrapper:
    '''
    dataset wrapper for counterfactually matched pairs in each batch
    '''
    def __init__(self, dataset, cf_ids, num_cf, use_raw_index=False):
        super().__init__()
        self.dataset = dataset
        self.cf_ids = cf_ids
        self.num_classes = len(self.cf_ids[0]) + 1
        print("num classes: ", self.num_classes)
        self.num_cf = num_cf
        print("num cf: ", self.num_cf)
        # ensure loading the correct cf examples for different splits
        self.use_raw_index = use_raw_index
        if use_raw_index and isinstance(self.dataset, misc._SplitDataset):
            self.raw_dataset = self.dataset.underlying_dataset
            self.idx_map = self.dataset.keys

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x_list = [x]
        y_list = [y]
        chosen_classes = random.sample(list(range(len(self.cf_ids[0]))), self.num_cf)
        
        if self.use_raw_index:
            for c in chosen_classes:
                cf_id = self.cf_ids[self.idx_map[index]][c]  

                if len(cf_id) > 0:
                    _cf_id = cf_id[0]
                    cf_x, cf_y = self.raw_dataset[_cf_id]

                    if cf_y == y:
                        print("factual example ", index, " and counterfactual example ", cf_id, " has the same label.")
                        exit(1)
                    elif cf_y < y:
                        assert cf_y == c
                    else:
                        assert cf_y == c + 1

                else:
                    _cf_id = None
                    cf_x, cf_y = torch.zeros_like(x, device=x.device), \
                        y - y - 1

                x_list.append(cf_x)
                y_list.append(cf_y)

        else:
            for c in chosen_classes:
                cf_id = self.cf_ids[index][c]

                if len(cf_id) > 0:
                    _cf_id = cf_id[0]
                    cf_x, cf_y = self.dataset[_cf_id]

                    if cf_y == y:
                        print("factual example ", index, " and counterfactual example ", cf_id, " has the same label.")
                        exit(1)
                    elif cf_y < y:
                        assert cf_y == c
                    else:
                        assert cf_y == c + 1

                else:
                    _cf_id = None
                    cf_x, cf_y = torch.zeros_like(x, device=x.device), \
                        y - y - 1

                x_list.append(cf_x)
                y_list.append(cf_y)

        return torch.stack(x_list, 0), torch.tensor(y_list)

    def __len__(self):
        return len(self.dataset)

def wrap_datasets(in_splits, matcher, num_classes, batch_size=32,
                dist_func='l2', device="cuda", verbose=True, use_raw_index=False, 
                threshold=False, topk=1, num_cf=1):
    '''
    wrap each dataset in the train split with counterfactually matched pairs
    '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    all_vars = [] # num_envs * num_classes list of tensors
    wrapped_datasets = []
    print("use ", dist_func, " distance.")
    for env, (dataset, weight) in enumerate(in_splits):
        print("processing environment ", env)
        x = [[] for i in range(num_classes)]
        ids = [[] for i in range(num_classes)]
        vars = []
        if verbose:
            print("total number of data: ", len(dataset))

        start_time = time.time()
        dataloader = DataLoader(IndexWrapper(dataset, use_raw_index), 
                batch_size=batch_size, shuffle=False, 
                num_workers=4, drop_last=False)
        y_counts = torch.zeros(num_classes, dtype=torch.int64, device=device)
        for index_batch, x_batch, y_batch in dataloader:
            index_batch_, x_batch_, y_batch_ = \
                copy.deepcopy(index_batch), copy.deepcopy(x_batch), copy.deepcopy(y_batch)
            del index_batch, x_batch, y_batch
            for i, x_, y_ in zip(index_batch_, x_batch_, y_batch_):
                y_counts[int(y_)] += 1
                x[y_].append(x_)
                ids[y_].append(i)
        del dataloader 
        print("data loading time: ", time.time() - start_time)

        x = [torch.stack(x_y, 0) for x_y in x]
        y = [torch.zeros(len(x[i]), dtype = torch.int64) + i for i in range(num_classes)]
        p_y = y_counts/torch.sum(y_counts)
        if verbose:
            print("x sizes: ", [x_y.size() for x_y in x])
            print("y sizes: ", [_y.size() for _y in y])
            print("p(y): ", p_y)

        start_time = time.time()
        matcher.eval()
        with torch.no_grad():
            for x_, y_ in zip(x,y):
                i = 0
                temp_vars = []
                while i < len(x_):
                    j = min(i+batch_size, len(x_))
                    temp_vars.append(matcher.propensity_score(x_[i:j].to(device),
                                    env, p_y).detach())
                    i = j
                temp_vars = torch.cat(temp_vars, 0)
                vars.append(temp_vars)
        matcher.train()
        print("latent variables inference time: ", time.time() - start_time)
        all_vars.append(vars)

        start_time = time.time()
        # calculate distance acorss different y
        most_simialr_id = {}
        for i in range(num_classes-1):
            for j in range(i+1, num_classes):
                if verbose:
                    print("vars {}: {}".format(i, vars[i]))
                    print("vars {}: {}".format(j, vars[j]))
                dist_ij = dist_mat(vars[i], vars[j], func=dist_func, device=device)
                if verbose:
                    print("dist_ij mat size: ", dist_ij.size())
                    print("dist_ij mat: ", dist_ij)
                if dist_func == 'kld':
                    dist_ji = dist_mat(vars[j], vars[i], func=dist_func, device=device)
                else:
                    dist_ji = torch.transpose(dist_ij, 0, 1)

                kj = min(topk, len(vars[j]))
                min_vals_ij, min_ids_ij = torch.topk(
                    dist_ij, k=kj, dim=1, largest=False)
                ki = min(topk, len(vars[i]))
                min_vals_ji, min_ids_ji = torch.topk(
                    dist_ji, k=ki, dim=1, largest=False)

                if threshold:
                    threshold_ij = torch.mean(dist_ij)
                    print("use threshold = ", threshold_ij.item())
                    mask_ij = min_vals_ij < threshold_ij

                    threshold_ji = torch.mean(dist_ji)
                    print("use threshold = ", threshold_ji.item())
                    mask_ji = min_vals_ji < threshold_ji

                    most_simialr_id[(i,j)] = []
                    for k in range(len(vars[i])):
                        most_simialr_id[(i,j)].append(min_ids_ij[k][mask_ij[k]].to('cpu'))

                    most_simialr_id[(j,i)] = []
                    for k in range(len(vars[j])):
                        most_simialr_id[(j,i)].append(min_ids_ji[k][mask_ji[k]].to('cpu'))
                    if verbose:
                        print("num matched ids: ", [len(a) for a in most_simialr_id[(i,j)]])
                        print("num matched ids: ", [len(a) for a in most_simialr_id[(j,i)]])
                else:
                    most_simialr_id[(i,j)], most_simialr_id[(j,i)] = \
                        min_ids_ij.to('cpu'), min_ids_ji.to('cpu')

                if verbose:
                    print("ki: ", ki)
                    print("kj: ", kj)
                    print("mins: ", min_vals_ij)
                    print("mins: ", min_vals_ji)

        # generate countefactual example ids for each class
        if use_raw_index and isinstance(dataset, misc._SplitDataset):
            cf_ids = [[] for _ in range(len(dataset.underlying_dataset))]
        else:
            cf_ids = [[] for _ in range(len(dataset))]
        for i in range(num_classes):
            for j in range(num_classes):
                if j != i:
                    for k in range(len(x[i])):
                        temp = []
                        for g in most_simialr_id[(i,j)][k]:
                            temp.append(ids[j][g])
                        cf_ids[ids[i][k]].append(temp)

        print("counterfactual matching time: ", time.time() - start_time)
        wrapped_datasets.append((CFmatchWrapper(dataset, cf_ids, num_cf, use_raw_index), weight))
        if verbose:
            print("weight: ", weight)
    if verbose:
        return wrapped_datasets, all_vars
    else:
        return wrapped_datasets

def oracle_wrap_CMNIST(in_splits, num_classes, num_color=2, batch_size=32,
                    verbose=True, use_raw_index=False, balance_rate=1.0):
    '''
    wrap each dataset in the train split with counterfactually matched pairs
    '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    wrapped_datasets = []
    print("num colors: ", num_color)
    for env, (dataset, weight) in enumerate(in_splits):
        print("processing environment ", env)
        ids = [[[] for i in range(num_classes)] for _ in range(num_color)]

        if verbose:
            print("total number of data: ", len(dataset))
        start_time = time.time()
        dataloader = DataLoader(IndexWrapper(dataset, use_raw_index), 
                batch_size=batch_size, shuffle=False, 
                num_workers=4, drop_last=False)
        for index_batch, x_batch, y_batch in dataloader:
            index_batch_, x_batch_, y_batch_ = \
                copy.deepcopy(index_batch), copy.deepcopy(x_batch), copy.deepcopy(y_batch)
            del index_batch, x_batch, y_batch
            for i, x_, y_ in zip(index_batch_, x_batch_, y_batch_):
                color_channel = 0
                for j in range(num_color):
                    if torch.sum(x_[j]) > 0:
                        ids[j][y_].append(i)
                        color_channel += 1
                if color_channel > 1:
                    print("more than one color channel")
        del dataloader 
        print("data loading time: ", time.time() - start_time)

        if verbose:
            print("ids sizes: ", [[len(x_y) for x_y in ids[i]] for i in range(num_color)])

        # generate countefactual example ids for each class
        if use_raw_index and isinstance(dataset, misc._SplitDataset):
            cf_ids = [[] for _ in range(len(dataset.underlying_dataset))]
        else:
            cf_ids = [[] for _ in range(len(dataset))]

        for i in range(num_color): # color
            for j in range(num_classes): # class
                for id in ids[i][j]:
                    for k in range(num_classes):
                        if k != j:
                            if bool(torch.rand(1) < balance_rate):
                                cf_ids[id].append(ids[i][k])
                            else:
                                cf_ids[id].append(ids[i][j])

        wrapped_datasets.append((CFmatchWrapper(dataset, cf_ids, num_classes-1,
                                 use_raw_index), weight))
        if verbose:
            print("weight: ", weight)
    if verbose:
        return wrapped_datasets, None
    else:
        return wrapped_datasets

### Eval metrics ###

def eval_mse(network, loader, d, weights, device):
    weights_offset = 0
    weighted_mse = 0.0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = len(x)
            recon_x = network.reconstruct(x, y, d)
            mse = torch.mean(F.mse_loss(recon_x.view(batch_size, -1), 
                    x.view(batch_size, -1), reduction='none'), -1)
            if weights is None:
                batch_weights = torch.ones(batch_size, device=device)
            else:
                batch_weights = weights[weights_offset : weights_offset + batch_size]
                weights_offset += batch_size
            batch_weights = batch_weights.to(device)
            weighted_mse += (torch.sum(mse * batch_weights)/torch.sum(batch_weights)).item()
    network.train()

    return weighted_mse/len(loader)

def eval_acc(network, loader, weights, num_classes, env, device):
    weights_offset = 0
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = len(x)
            z = network.get_z(x, y, env)
            mse_list = []
            for i in range(num_classes):
                l = torch.zeros(len(x), dtype = torch.int64, device=device) + i
                result = network.inference(z, l)
                mse = torch.mean(F.mse_loss(result['recon_x'].view(batch_size, -1), 
                        x.view(batch_size, -1), reduction='none'), -1)
                mse_list.append(mse)
            pred_y = torch.argmin(torch.stack(mse_list, 1), 1)
            
            if weights is None:
                batch_weights = torch.ones(batch_size, device=device)
            else:
                batch_weights = weights[weights_offset : weights_offset + batch_size]
                weights_offset += batch_size
            correct += ((y == pred_y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

### Jupyter Notebook functions ###

def load_model(model_name, checkpoint_path, device):
    matcher_class = algorithms.get_algorithm_class(model_name)
    matcher_checkpoint = torch.load(checkpoint_path)
    args = matcher_checkpoint["args"]
    hparams = hparams_registry.default_hparams(args['algorithm'], args['dataset'])
    hparams.update(matcher_checkpoint["model_hparams"])
    print('hparams: ', hparams)
    print('args: ', args)

    if args['dataset'] in vars(datasets):
        dataset = vars(datasets)[args['dataset']](args['data_dir'],
            args['test_envs'], hparams)
    else:
        raise NotImplementedError

    matcher = matcher_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args['test_envs']), hparams)
    matcher.load_state_dict(matcher_checkpoint['model_dict'])
    matcher.to(device)

    return matcher, dataset, hparams, args

def process_img(img, dataset_name, hparams):
    if dataset_name == 'ColoredMNIST':
        img = torch.cat([img.view(2,28,28), torch.zeros([1,28,28], device=img.device)], 0)
    elif dataset_name == 'ColoredMNIST_10class':
        img = img.view(10,28,28)
    elif dataset_name in ["VLCS",
                        "PACS",
                        "OfficeHome",
                        "TerraIncognita",
                        "DomainNet",
                        "SVIRO"]:
        img = img.view(3, hparams['size'], hparams['size'])
        if hparams["normalize"]:
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).unsqueeze(-1).unsqueeze(-1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device).unsqueeze(-1).unsqueeze(-1)
            img = img * std + mean
    return img


def visualize(dataset_name, splits_match, env, idx, num_classes, hparams, figsize=5):
    dataset = splits_match[env][0]
    imgs = [] 
    ys = []
    for i in range(num_classes-1):
        x, y, cf_x, cf_y, cf_id = dataset.get_cf_example(idx, i, verbose=True)
        print(cf_id)
        if i == y:
            imgs.append(process_img(x, dataset_name, hparams))
            ys.append(y)
        imgs.append(process_img(cf_x, dataset_name, hparams))
        ys.append(cf_y)
    if y == num_classes - 1:
        imgs.append(process_img(x, dataset_name, hparams))
        ys.append(y)
    num_imgs = len(imgs)
    
    plt.figure()
    plt.figure(figsize=(figsize*num_classes, figsize))
    for p in range(num_imgs):
        plt.subplot(1, num_imgs, p+1)
        if isinstance(ys[p], torch.Tensor):
            _y = ys[p].item()
        else:
            _y = ys[p]
        if p == y:
            plt.title("y={:d}".format(_y), color='red', fontsize=10*figsize)
        else:
            plt.title("y={:d}".format(_y), color='black', fontsize=10*figsize)

        if len(imgs[p])<=3:
            plt.imshow(imgs[p].permute(1,2,0).cpu().data.numpy())
        else:
            color_palette = np.array([[144, 157, 94], [43, 242, 126], 
                    [14, 81, 175], [40, 126, 94], [45, 16, 82], [193, 144, 168], 
                    [231, 85, 17], [191, 84, 52], [251, 140, 192], [234, 67, 90]])/255
            img = np.matmul(imgs[p].permute(1,2,0), color_palette)
            img = 1-img.cpu().data.numpy()
            plt.imshow(img)
        plt.axis('off')

def visualize_data(dataset_name, splits_match, env, hparams, num_imgs=10, figsize=5):
    dataset = splits_match[env][0].dataset
    imgs = [] 
    ys = []
    for i in range(num_imgs):
        x,y = dataset[random.randint(0, 8000)]
        imgs.append(process_img(x, dataset_name, hparams))
        ys.append(y)
    
    plt.figure()
    plt.figure(figsize=(figsize*10, figsize))
    for p in range(num_imgs):
        plt.subplot(1, num_imgs, p+1)
        if isinstance(ys[p], torch.Tensor):
            _y = ys[p].item()
        else:
            _y = ys[p]
        plt.title("y={:d}".format(_y), color='black', fontsize=10*figsize)

        if len(imgs[p])<=3:
            plt.imshow(imgs[p].permute(1,2,0).cpu().data.numpy())
        else:
            color_palette = np.array([[144, 157, 94], [43, 242, 126], 
                    [14, 81, 175], [40, 126, 94], [45, 16, 82], [193, 144, 168], 
                    [231, 85, 17], [191, 84, 52], [251, 140, 192], [234, 67, 90]])/255
            img = np.matmul(imgs[p].permute(1,2,0), color_palette)
            img = 1-img.cpu().data.numpy()
            plt.imshow(img)
        plt.axis('off')

def reconstruct(dataset_name, splits_match, network, env, idx, hparams, device):
    dataset = splits_match[env][0].dataset
    x, y = dataset[idx]
    x = x.unsqueeze(0).to(device)
    y = torch.tensor([y]).to(device)
    network.eval()
    with torch.no_grad():
        recon_x = network.reconstruct(x, y, env)
    network.train()
    img = process_img(x.squeeze(), dataset_name, hparams)
    recon_img = process_img(recon_x.squeeze(), dataset_name, hparams)
    plt.figure()
    plt.figure(figsize=(20, 10))
    if isinstance(y[0], torch.Tensor):
        print("label: ", y[0].item())
    else:
        print("label: ", y[0])
    
    plt.subplot(1, 2, 1)
    if len(img)<=3:
        plt.imshow(img.permute(1,2,0).cpu().data.numpy())
    else:
        color_palette = np.array([[144, 157, 94], [43, 242, 126], 
                [14, 81, 175], [40, 126, 94], [45, 16, 82], [193, 144, 168], 
                [231, 85, 17], [191, 84, 52], [251, 140, 192], [234, 67, 90]])/255
        img = np.matmul(img.permute(1,2,0).cpu().data.numpy(), color_palette)
        img = 1-img
        plt.imshow(img)

    plt.subplot(1, 2, 2)
    if len(recon_img)<=3:
        plt.imshow(recon_img.permute(1,2,0).cpu().data.numpy())
    else:
        color_palette = np.array([[144, 157, 94], [43, 242, 126], 
                [14, 81, 175], [40, 126, 94], [45, 16, 82], [193, 144, 168], 
                [231, 85, 17], [191, 84, 52], [251, 140, 192], [234, 67, 90]])/255
        recon_img = np.matmul(recon_img.permute(1,2,0).cpu().data.numpy(), color_palette)
        recon_img = 1-recon_img
        plt.imshow(recon_img)
    plt.axis('off')

def intervene(dataset_name, splits_match, network, env, idx, device, 
                num_classes, hparams, figsize=5):
    try:
        dataset = splits_match[env][0].dataset
    except:
        dataset = splits_match[env][0]
    x, y = dataset[idx]
    imgs = [process_img(x, dataset_name, hparams)]
    x = x.unsqueeze(0).to(device)
    if isinstance(y, torch.Tensor):
        y = y.item()
    ys = [y]
    y = torch.zeros(1, dtype = torch.int64, device=device) + y
    network.eval()
    with torch.no_grad():
        z = network.get_z(x, y, env)
        for i in range(num_classes):
            img = network.inference(z, torch.tensor([i], device=device))['recon_x']
            imgs.append(process_img(img.squeeze(), dataset_name, hparams))
            ys.append(i)
    network.train()
    num_imgs = len(imgs)
    
    plt.figure()
    plt.figure(figsize=(figsize*(num_classes+1), figsize))
    for p in range(num_imgs):
        plt.subplot(1, num_imgs, p+1)
        if isinstance(ys[p], torch.Tensor):
            _y = ys[p].item()
        else:
            _y = ys[p]
        if p == y:
            plt.title("y={:d}".format(_y), color='red', fontsize=10*figsize)
        else:
            plt.title("y={:d}".format(_y), color='black', fontsize=10*figsize)

        if len(imgs[p])<=3:
            plt.imshow(imgs[p].permute(1,2,0).cpu().data.numpy())
        else:
            color_palette = np.array([[144, 157, 94], [43, 242, 126], 
                    [14, 81, 175], [40, 126, 94], [45, 16, 82], [193, 144, 168], 
                    [231, 85, 17], [191, 84, 52], [251, 140, 192], [234, 67, 90]])/255
            img = np.matmul(imgs[p].permute(1,2,0).cpu().data.numpy(), color_palette)
            img = 1-img
            plt.imshow(img)
        plt.axis('off')