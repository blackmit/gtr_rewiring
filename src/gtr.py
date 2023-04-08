import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform

import networkx as nx
from functools import reduce

def get_component_mask(
    graph : nx.Graph
) -> torch.Tensor:
    """ Return a n x n torch tensor where the i,j entry is 1 iff i and j are in the same connected component of `graph` """
    num_vertices = graph.number_of_nodes()

    def single_component_mask(
        vertices_of_component : list
    ) -> torch.Tensor:
        """ Return a binary matrix where each i,j entry is 1 iff i and j are in `vertices_of_component` """
        row_mask = torch.zeros((num_vertices, num_vertices))
        col_mask = torch.zeros((num_vertices, num_vertices))
        row_mask[vertices_of_component,:] = 1
        col_mask[:,vertices_of_component] = 1
        return torch.logical_and(row_mask, col_mask)

    component_mask = reduce(
        torch.logical_or,
        [single_component_mask(list(component)) for component in nx.connected_components(graph)]
    )
    return component_mask


def compute_edges(
    data : torch_geometric.data.Data,
    num_edges : int,
    try_gpu : bool = True
) -> torch.Tensor:
    """ Calculate edges to add to the graph using the GTR algorithm. """
    # retrieve device used for matrix computations
    if try_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # prepare the pseudoinverse and squared pseudoinverse
    graph = torch_geometric.utils.convert.to_networkx(data).to_undirected()
    laplacian = torch.tensor(nx.laplacian_matrix(graph).todense().astype("float")).to(device)
    pinv = torch.linalg.pinv(laplacian, hermitian=True)
    squared_pinv = pinv @ pinv
    # get component mask
    component_mask = get_component_mask(graph).to(device)
    # Compute num_edges_to_add edges and append them to gtr_edges
    gtr_edges = torch.zeros((2,0), dtype=torch.long)
    for _ in range(num_edges):
        # The entries resistance_matrix[s,t] and biharmonic_matrix[s,t]
        # are the effective resistance and biharmonic distance between s and t.
        pinv_diagonal = torch.diagonal(pinv)
        resistance_matrix = pinv_diagonal.unsqueeze(0) + pinv_diagonal.unsqueeze(1) - 2*pinv
        squared_pinv_diagonal = torch.diagonal(squared_pinv)
        biharmonic_matrix = squared_pinv_diagonal.unsqueeze(0) + squared_pinv_diagonal.unsqueeze(1) - 2*squared_pinv
        # diff_matrix[s,t] stores the change in total resistance when the edge {s,t} is added to the graph
        diff_matrix = (biharmonic_matrix / (1 + resistance_matrix))
        # We only want to add an edge not already in the graph and not between connected components.
        # Multiplying by edge_mask sets the value of all self-loops
        # and edges already in the graph to 0, as these entries
        # are the only non-zero entries in the laplacian.
        # Multiplying by component_mask sets the value
        # of all edges between connected components to 0.
        edge_mask = torch.logical_not(laplacian.bool())
        masked_diff_matrix = diff_matrix * edge_mask * component_mask
        # Find the endpoints of the edge that most decrease the total resistance.
        # torch.argmax returns the index of max in flattened coordinates, hence the divmod.
        s, t = divmod(torch.argmax(masked_diff_matrix).cpu().item(), data.num_nodes)
        # Add the edge {s,t} to the return array
        new_edges = torch.Tensor([[s, t], [t, s]]).long()
        gtr_edges = torch.cat([gtr_edges, new_edges], 1)
        # Update the pseudoinverse with Woodbury's formula
        v = pinv[:,s] - pinv[:,t]
        effective_resistance = resistance_matrix[s,t]
        pinv = pinv - (1/(1+effective_resistance))*torch.outer(v, v)
        # Update the squared pseudoinverse with Woodbury's formula
        x = torch.zeros(data.num_nodes).to(device)
        x[s], x[t] = 1, -1
        y = laplacian[:,s] - laplacian[:,t]
        U = torch.column_stack([x, y+x])
        V = torch.stack([y+x, x])
        left = squared_pinv @ U
        center = torch.inverse(torch.eye(2) + V@squared_pinv@U)
        right = V @ squared_pinv
        squared_pinv = squared_pinv - left@center@right
        # update the laplacian
        laplacian[s,s] += 1; laplacian[s,t] -= 1
        laplacian[t,s] -= 1; laplacian[t,t] += 1
    return gtr_edges


@functional_transform('add_gtr_edges')
class AddGTREdges(BaseTransform):
    def __init__(
        self,
        num_edges : int,
        try_gpu : bool = True
    ):
        print(
            "We do not recommend using AddGTREdges.",
            "AddGTREdges computes additional edges each time a graph is loaded,",
            "which is computationally expensive.",
            "We recommend using PrecomputeGTREdges and AddPrecomputedGTREdges instead."
        )
        self.num_edges = num_edges
        self.try_gpu = try_gpu

    def __call__(
        self,
        data : torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        # add edge types
        if hasattr(data, "edge_type"):
            # if data already has edge types, label new edges with next unused int
            new_edge_type = torch.max(data.edge_type) + 1
            data.edge_type = torch.cat([
                data.edge_type,
                torch.full((2*self.num_edges,), new_edge_type, dtype=torch.long)
            ])
        else:
            # if data does not already have edge types, label original edges 0 and new edges 1
            num_original_edges = data.edge_index.shape[1]
            data.edge_type = torch.cat([
                torch.zeros(num_original_edges, dtype=torch.long),
                torch.ones(2*self.num_edges, dtype=torch.long)
            ])
        # add edges
        new_edges = compute_edges(data, self.num_edges, self.try_gpu)
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        return data


@functional_transform('add_precomputed_gtr_edges')
class AddPrecomputedGTREdges(BaseTransform):
    def __init__(
        self,
        num_edges : int
    ):
        self.num_edges = num_edges

    def __call__(
        self,
        data : torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        if not hasattr(data, "precomputed_gtr_edges"):
            raise AttributeError("Data does not have precomputed edges. Edges must be computed with the pre_transform PrecomputeGTREdges.")
        if data.precomputed_gtr_edges.shape[1] < 2*self.num_edges:
            raise ValueError("Too few edges have been precomputed.")
        # add edge types
        if hasattr(data, "edge_type"):
            # if data already has edge types, label new edges with next unused int
            new_edge_type = torch.max(data.edge_type) + 1
            data.edge_type = torch.cat([
                data.edge_type,
                torch.full((2*self.num_edges,), new_edge_type, dtype=torch.long)
            ])
        else:
            # if data does not already have edge types, label original edges 0 and new edges 1
            num_original_edges = data.edge_index.shape[1]
            data.edge_type = torch.cat([
                torch.zeros(num_original_edges, dtype=torch.long),
                torch.ones(2*self.num_edges, dtype=torch.long)
            ])
        # add new edges
        new_edges = data.precomputed_gtr_edges[:,:2*self.num_edges]
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        return data

@functional_transform('precompute_gtr_edges')
class PrecomputeGTREdges(BaseTransform):
    def __init__(
        self,
        num_edges : int,
        try_gpu : bool = True
    ):
        self.num_edges = num_edges
        self.try_gpu = try_gpu

    def __call__(
        self,
        data : torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        data.precomputed_gtr_edges = compute_edges(data, self.num_edges, self.try_gpu)
        return data
