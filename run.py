def train_predict():
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    affinity_mat = load_data(args.dataset)
    train_data, test_data, num_drug, num_target = process_data(affinity_mat, args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    drug_graphs_dict = get_drug_molecule_graph(json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    drug_graphs_Data = DrugGraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate_seq,batch_size=num_drug)

    target_graphs_dict = get_target_molecule_graph(json.load(open(f'data/{args.dataset}/targets_pdb.txt'), object_pairs_hook=OrderedDict), args.dataset)
    target_graphs_Data = ProtGraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate_seq,batch_size=num_target)

    drug_seqs_dict = get_drug_molecule_sequence(json.load(open(f'data/{args.dataset}/drugs.txt'),object_pairs_hook=OrderedDict),args.dataset)
    drug_seqs_Data = SeqDataset(seq_dict=drug_seqs_dict, dttype="drug")
    drug_seqs_DataLoader = torch.utils.data.DataLoader(drug_seqs_Data, shuffle=False, collate_fn=collate_seq, batch_size=num_drug)

    target_seqs_dict = get_target_molecule_sequence(json.load(open(f'data/{args.dataset}/targets_pdb.txt'), object_pairs_hook=OrderedDict),args.dataset)
    target_seqs_Data = SeqDataset(seq_dict=target_seqs_dict, dttype="target")
    target_seqs_DataLoader = torch.utils.data.DataLoader(target_seqs_Data, shuffle=False, collate_fn=collate_seq,batch_size=num_target)

    print("Model preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = PredictModule()

    model.to(device)

    print("Start training...")

    for epoch in range(args.epochs):
        train(model, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, drug_seqs_DataLoader, target_seqs_DataLoader, args.lr, epoch+1, args.batch_size)
        G, P = test(model, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader,drug_seqs_DataLoader, target_seqs_DataLoader)
        r = model_evaluate(G, P)
        print(r)

    print('\npredicting for test data')

    G, P = test(model, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, drug_seqs_DataLoader, target_seqs_DataLoader)
    result = model_evaluate(G, P)
    print("result:", result)


def train(model, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, drug_seqs_DataLoader, target_seqs_DataLoader,
          lr, epoch, batch_size):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0)

    drug_graph_batchs = [graph.to(device) for batch in drug_graphs_DataLoader for graph in batch]
    target_graph_batchs = [graph.to(device) for batch in target_graphs_DataLoader for graph in batch]
    drug_seq_batchs = [seq.to(device) for batch in drug_seqs_DataLoader for seq in batch]
    target_seq_batchs = [seq.to(device) for batch in target_seqs_DataLoader for seq in batch]

    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        drug_graph = [Batch.from_data_list([drug_graph_batchs[i] for i in data.drug_id])]
        target_graph = [Batch.from_data_list([target_graph_batchs[i] for i in data.target_id])]
        drug_seq = [drug_seq_batchs[i] for i in data.drug_id]
        target_seq = [target_seq_batchs[i] for i in data.target_id]
        output = model(drug_graph, target_graph, drug_seq, target_seq)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

    print(f"Epoch {epoch} completed. Average Loss: {(total_loss / len(train_loader)):.6f}")


def test(model, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, drug_seqs_DataLoader, target_seqs_DataLoader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = [graph.to(device) for batch in drug_graphs_DataLoader for graph in batch]
    target_graph_batchs = [graph.to(device) for batch in target_graphs_DataLoader for graph in batch]
    drug_seq_batchs = [seq.to(device) for batch in drug_seqs_DataLoader for seq in batch]
    target_seq_batchs = [seq.to(device) for batch in target_seqs_DataLoader for seq in batch]

    with torch.no_grad():
        for data in loader:
            drug_graph = [Batch.from_data_list([drug_graph_batchs[i] for i in data.drug_id])]
            target_graph = [Batch.from_data_list([target_graph_batchs[i] for i in data.target_id])]
            drug_seq = [drug_seq_batchs[i] for i in data.drug_id]
            target_seq = [target_seq_batchs[i] for i in data.target_id]
            output = model(drug_graph, target_graph, drug_seq, target_seq)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()



if __name__ == '__main__':
    import os
    import argparse
    import torch
    import json
    import warnings
    from collections import OrderedDict
    from torch import nn
    from data_process import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph, get_drug_molecule_sequence, get_target_molecule_sequence
    from utils import DrugGraphDataset, ProtGraphDataset, SeqDataset, collate, collate_seq, model_evaluate
    from model import PredictModule
    from torch_geometric.data import Batch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    args, _ = parser.parse_known_args()

    train_predict()




