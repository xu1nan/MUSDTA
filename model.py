import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import AttentiveFP


class ProtGCNBlock(nn.Module):
    def __init__(self, php_emb=33, dihedrals=6, esm_emb=2560, edge=39, gcn=[128, 256, 256]):
        super().__init__()
        # Feature projection layers
        self.proj_node = nn.Linear(php_emb + dihedrals + esm_emb, 128)
        self.proj_edge = nn.Linear(edge, gcn[0])

        # GCN layer construction
        gcn_dims = [gcn[0]] + gcn
        gcn_layers = []
        for in_dim, out_dim in zip(gcn_dims[:-1], gcn_dims[1:]):
            gcn_layers.append((
                torch_geometric.nn.TransformerConv(
                    in_dim, out_dim, edge_dim=gcn[0]),
                'x, edge_index, edge_attr -> x'
            ))
            gcn_layers.append(nn.LeakyReLU())

        self.gcn = torch_geometric.nn.Sequential('x, edge_index, edge_attr', gcn_layers)
        self.pool = torch_geometric.nn.global_mean_pool

    def forward(self, seq, edge_index, node_s, esm_emb, edge_s, batch):
        # Feature concatenation
        x = torch.cat([seq, node_s, esm_emb], dim=-1)
        x = self.proj_node(x)

        # Edge feature processing
        edge_attr = self.proj_edge(edge_s)

        # Graph convolution
        x = self.gcn(x, edge_index, edge_attr)
        return self.pool(x, batch)


class ProtGCNModel(nn.Module):
    def __init__(self, pretrained_emb, gcn):
        super().__init__()
        self.graph_conv = ProtGCNBlock(esm_emb=pretrained_emb, gcn=gcn)

    def forward(self, graph_batchs):
        return torch.cat([self.graph_conv(
            graph.seq,
            graph.edge_index,
            graph.node_s,
            graph.esm_emb,
            graph.edge_s,
            graph.batch
        ) for graph in graph_batchs], dim=0)


class AttentionFPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, num_layers, num_timesteps, dropout_rate):
        super().__init__()
        self.attentive_fp = AttentiveFP(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            edge_dim=edge_dim,
            dropout=dropout_rate
        )

    def forward(self, x, edge_index, edge_attr, x_batch):
        return self.attentive_fp(x, edge_index, edge_attr, batch=x_batch)


class DrugGCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, num_layers, num_timesteps, dropout_rate):
        super().__init__()
        self.attention_block = AttentionFPBlock(
            input_dim, hidden_dim, output_dim, edge_dim,
            num_layers, num_timesteps, dropout_rate
        )

    def forward(self, graph_batchs):
        return torch.cat([
            self.attention_block(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch
            ) for graph in graph_batchs
        ], dim=0)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.activation(self.conv(x)))


class CNNModel(nn.Module):
    """1D CNN feature extractor"""

    def __init__(self, input_channels, dropout_rate):
        super().__init__()
        self.layers = nn.Sequential(
            CNNBlock(input_channels, 1024, 1, dropout_rate),
            CNNBlock(1024, 512, 1, dropout_rate),
            CNNBlock(512, 256, 1, dropout_rate)
        )

    def forward(self, batch_data):
        embeddings = []
        for data in batch_data:
            x = data.x.unsqueeze(0).permute(0, 2, 1)
            x = self.layers(x)
            embeddings.append(x.mean(dim=2))
        return torch.cat(embeddings, dim=0)


class PredictModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_graph_conv = DrugGCNModel(18, 64, 16, 12, 3, 3, 0.3)
        self.target_graph_conv = ProtGCNModel(2560, [128, 256, 256])
        self.drug_seq_conv = CNNModel(768, 0.1)
        self.target_seq_conv = CNNModel(2560, 0.1)
        self.cross_attention = nn.MultiheadAttention(256, 1)
        self.mlp = nn.Sequential(
            nn.Linear(16, 256),
            nn.Linear(256 * 4, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, drug_graph_batchs, target_graph_batchs, drug_seq_batchs, target_seq_batchs):
        drug_graph = self.mlp[0](self.drug_graph_conv(drug_graph_batchs)).unsqueeze(1)
        target_graph = self.target_graph_conv(target_graph_batchs).unsqueeze(1)
        drug_seq = self.drug_seq_conv(drug_seq_batchs).unsqueeze(1)
        target_seq = self.target_seq_conv(target_seq_batchs).unsqueeze(1)

        drug_features = torch.cat([drug_graph, drug_seq], dim=1).permute(1, 0, 2)
        target_features = torch.cat([target_graph, target_seq], dim=1).permute(1, 0, 2)
        drug_att, _ = self.cross_attention(drug_features, drug_features, drug_features)
        target_att, _ = self.cross_attention(target_features, target_features, target_features)

        fused_features = torch.cat([drug_att, target_att], dim=0)
        fused_att, _ = self.cross_attention(fused_features, fused_features, fused_features)

        features = [
            0.5 * fused_att[:drug_graph.size(1)] + 0.5 * drug_graph.permute(1, 0, 2),
            0.5 * fused_att[drug_graph.size(1):drug_graph.size(1) + drug_seq.size(1)] + 0.5 * drug_seq.permute(1, 0, 2),
            0.5 * fused_att[drug_graph.size(1) + drug_seq.size(1):-target_seq.size(1)] + 0.5 * target_graph.permute(1,0,2),
            0.5 * fused_att[-target_seq.size(1):] + 0.5 * target_seq.permute(1, 0, 2)
        ]

        pooled = []
        for f in features:
            f = f.permute(1, 2, 0)
            pooled_feature = self.pool(f).squeeze(-1)
            pooled.append(pooled_feature)

        return self.mlp[1:](torch.cat(pooled, dim=1))
