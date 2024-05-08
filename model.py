import torch
from torch_geometric.nn import GCNConv, SAGPooling, MessagePassing, GATConv, RGCNConv, RGATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import Linear, Sequential, ReLU, Sigmoid
from torch_geometric.utils import degree, add_self_loops
from torch_scatter import scatter
import torch.nn.functional as F
from torch_scatter.utils import broadcast


class gcnConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(gcnConv, self).__init__(aggr='add')
        self.linear1 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.mlp = Sequential(
            Linear(6, 64),
            ReLU(),
            Linear(64, 256),
            ReLU(),
            Linear(256, 512),
            ReLU(),
            Linear(512, 1024),
        )
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(x, edge_index, edge_attr)

    def message(self, x, edge_index, edge_attr):
        x = self.linear1(x)
        edgeWeight = self.mlp(edge_attr)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x_j = x[row]
        x_j = norm.view(-1, 1) * x_j
        return x_j, edgeWeight

    def aggregate(self, x_j, edge_index, edgeWeight):
        row, col = edge_index
        index = broadcast(col, x_j, 0)
        size = list(x_j.size())
        if index.numel() == 0:
            size[0] = 0
        else:
            size[0] = int(index.max()) + 1
        out = torch.zeros(size, dtype=x_j.dtype, device=x_j.device)

        out = out.scatter_add_(0, index, x_j+edgeWeight)
        return out

    def update(self, x, out):
        return out + x


    def propagate(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        zeros = torch.zeros(edge_index.shape[1] - edge_attr.shape[0], 6, dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, zeros])

        out, edgeWeight = self.message(x, edge_index, edge_attr)
        out = self.aggregate(out, edge_index, edgeWeight)
        out = self.update(x, out)

        return out

class mGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(mGCN, self).__init__()
        self.conv1 = gcnConv(num_node_features, num_node_features)
        self.conv2 = gcnConv(num_node_features, num_node_features)
        self.conv3 = gcnConv(num_node_features, num_node_features)
        self.pool = SAGPooling(num_node_features*3, ratio=100)
        self.mlp = Sequential(
            Linear(num_node_features*6, 1024),
            ReLU(),
            Linear(1024, 256),
            ReLU(),
            Linear(256, 64),
            ReLU(),
            Linear(64, 16),
            ReLU(),
            Linear(16, num_classes)
        )


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        layer1 = self.conv1(x, edge_index, edge_attr).relu()

        layer2 = self.conv2(layer1, edge_index, edge_attr).relu()
        layer3 = self.conv3(layer2, edge_index, edge_attr).relu()

        layer = torch.cat([layer1, layer2, layer3], dim=1)

        pool, _, _, batch, _, _ = self.pool(layer, edge_index, batch=batch)

        readout = torch.cat([global_mean_pool(pool, batch), global_max_pool(pool, batch)], dim=1)
        readout = F.dropout(readout, p=0.5, training=self.training)

        logits = self.mlp(readout)

        return logits

