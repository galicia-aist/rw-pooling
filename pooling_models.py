import torch
from torch_geometric.nn import GCNConv, TopKPooling
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

class HierarchicalGCN_TOPK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_TOPK, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.9)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.9)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)