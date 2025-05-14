import torch
import numpy as np
from scipy import stats
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA


class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, drug_ids=None, target_ids=None, y=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.process(drug_ids, target_ids, y)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, drug_ids, target_ids, y):
        data_list = []
        for i in range(len(drug_ids)):
            DTA = DATA.Data(drug_id=torch.IntTensor([drug_ids[i]]), target_id=torch.IntTensor([target_ids[i]]),
                            y=torch.FloatTensor([y[i]]))
            data_list.append(DTA)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ProtGraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(ProtGraphDataset, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for key in graphs_dict:
            size, X_ca, seq, node_s, node_v, edge_s, edge_v, edge_index, esm_seq = graphs_dict[key]
            GVPData = DATA.Data(x=X_ca, seq=torch.Tensor(seq),
                                node_s=node_s, node_v=node_v,
                                edge_s=edge_s, edge_v=edge_v,
                                edge_index=edge_index, esm_emb=torch.Tensor(esm_seq))
            GVPData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            data_list.append(GVPData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DrugGraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(DrugGraphDataset, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for key in graphs_dict:
            size, features, edge_attr, edge_index = graphs_dict[key]
            graph_data = DATA.Data(x=torch.Tensor(features), edge_attr=torch.Tensor(edge_attr),
                                   edge_index=torch.LongTensor(edge_index))
            graph_data.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            data_list.append(graph_data)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SeqDataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, seq_dict=None, dttype=None):
        super(SeqDataset, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(seq_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, seq_dict):
        data_list = []
        for key in seq_dict:
            size, seq_feature = seq_dict[key]
            CNNData = DATA.Data(x=torch.Tensor(seq_feature))
            CNNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            data_list.append(CNNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


def collate_seq(data_list):
    return data_list


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)
    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / down)


def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)
    return CI


def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def model_evaluate(Y, P):
    return (get_mse(Y, P),
            get_rm2(Y, P),
            get_cindex(Y, P),
            get_pearson(Y, P),
            get_spearman(Y, P))





