from dgl.sparse import SparseMatrix
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from dgl.nn.pytorch.conv import GraphConv
import dgl
import numpy
import scipy.sparse
from . import BaseModel, register_model

class HeteroGraphConvLayer(nn.Module):
    """
    Multiplex Heterograph Convolution Layer
    
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self,in_feat,out_feat):
        super(HeteroGraphConvLayer,self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        # self.rel_names = rel_names

        # self.conv = dgl.nn.pytorch.HeteroGraphConv({
        #     rel:dgl.nn.pytorch.GraphConv(in_feat,out_feat)
        #     for rel in rel_names
        # })
        self.conv = dgl.nn.pytorch.HeteroGraphConv({"aggregated_relation":dgl.nn.pytorch.GraphConv(in_feat,out_feat,weight=True,bias=True)})
    
    def forward(self,hg,inputs):
        """Forward computation
        Parameters
        ----------
        hg : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        hs = self.conv(hg, inputs)


        return hs

class GraphConvLayer(nn.Module):
    """
    Multiplex Heterograph Convolution Layer
    
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self,in_feat,out_feat):
        super(GraphConvLayer,self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.conv =dgl.nn.pytorch.GraphConv(in_feat,out_feat,weight=True,bias=True)
    
    def forward(self,g,inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : torch.Tensor
            Node feature (adjacency matrix) for each node.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        hs = self.conv(g, inputs)


        return hs
@register_model('MHGCN')
class MHGCN(BaseModel):
    """
    **Title:** `Multiplex Heterogeneous Graph Convolutional Network <https://doi.org/10.1145/3534678.3539482>`_

    **Authors:** Pengyang Yu, Chaofan Fu, Yanwei Yu, Chao Huang, Zhongying Zhao, Junyu Dong.

    Parameters
    ----------
    in_dim : int
        Input feature size.
    out_dim : int
        Output feature size.
    num_hidden_layers: int
        Number of hidden GCN layers
    dropout : float, optional
        Dropout rate. Default: 0.0

    """
    @classmethod 
    def build_model_from_args(cls,args,hg):
        device = torch.device('cpu')
        if(hasattr(args,'gpu') and args.gpu != -1):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return cls(args.hidden_dim,
                   args.out_dim,
                   hg,
                   device,
                   args.use_hgcn,
                   args.num_layers - 2,
                   dropout=args.dropout)


    def __init__(self, 
                 in_dim,
                 out_dim,
                 hg,
                 device,
                 use_hgcn,
                 num_hidden_layers=1,
                 dropout=0):
        super(MHGCN,self).__init__()
        # init parameters
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_hgcn = use_hgcn


        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

        self.weight_b = torch.nn.Parameter(torch.FloatTensor(len(hg.etypes), 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        # input layer
        self.layers = nn.ModuleList()
        if(not self.use_hgcn):
            conv = GraphConvLayer(self.in_dim,out_dim)
            self.layers.append(conv)
            # hidden dimentions
            for index in range(num_hidden_layers):
                conv = GraphConvLayer(out_dim,out_dim)
                self.layers.append(conv)
            # output demention
            conv = GraphConvLayer(out_dim,out_dim)
            self.layers.append(conv)
        else:
            conv = HeteroGraphConvLayer(self.in_dim,out_dim)
            self.layers.append(conv)
            # hidden dimentions
            for index in range(num_hidden_layers):
                conv = HeteroGraphConvLayer(out_dim,out_dim)
                self.layers.append(conv)
            # output demention
            conv = HeteroGraphConvLayer(out_dim,out_dim)
            self.layers.append(conv)

        # a map for node type name ->  start index of adj
        self.node_index_map = dict()
        # a map for node type name -> count of nodes
        self.node_num_map = dict()
        node_index = 0
        for ntype in hg.ntypes:
            self.node_index_map[ntype] = node_index
            self.node_num_map[ntype] = hg.num_nodes(ntype)
            node_index += hg.num_nodes(ntype)

    

    def _multiplex_relation_aggregation(self,hg:dgl.DGLGraph) -> torch.Tensor:
        r"""
        We first generate multiple sub-graphs by differentiating the types of 
        edge connections between nodes in the multiplex and heterogeneous graph. 
        Afterwards, we aggregate the relation-aware graph contextual information 
        with different importance weights.

        Parameters
        ----------
        hg: dgl.HeteroGraph
            Input graph
        Returns
        -------
        feature:torch.Tensor
            adjacency matrix that aggregated by multiplex relation
        """
        A_t_list = []

        N = hg.num_nodes() - 1

        # generate multiple sub-graphs by differentiating the types of edge connections
        for canonical_etype in hg.canonical_etypes:
            src_node_type,etype,dst_node_type = canonical_etype
            sparse_matrix = hg.adjacency_matrix(canonical_etype)
            
            end_point = torch.Tensor([[N ],[N]]).to(self.device)


            sparse_matrix_indices = sparse_matrix.indices()
            sparse_matrix_indices[0] += self.node_index_map[src_node_type]
            sparse_matrix_indices[1] += self.node_index_map[dst_node_type]
            sparse_matrix_val = sparse_matrix.val
            if(sparse_matrix_indices[0].max() < N or sparse_matrix_indices[1].max() < N):
                sparse_matrix_indices = torch.cat((sparse_matrix_indices,end_point),1)
                sparse_matrix_indices = sparse_matrix_indices.long()
                sparse_matrix_val = torch.cat((sparse_matrix_val,torch.Tensor([0]).to(self.device)),0)


            sparse_matrix_shape = (N+1,N+1)
            new_sparse_matrix = SparseMatrix(torch.ops.dgl_sparse.from_coo(sparse_matrix_indices, sparse_matrix_val, sparse_matrix_shape))

            # sparse_matrix
            sparse_tensor = new_sparse_matrix.to_dense()

            A_t_list.append(sparse_tensor)
        
        # multiplex aggregation 
        A_t = torch.stack(A_t_list,dim=2).to_dense()

        # bias
        aggr_with_bias = torch.matmul(A_t,self.weight_b)

        # resize dim to 2
        aggr_with_bias = torch.squeeze(aggr_with_bias,2)
        aggr_with_bias = aggr_with_bias + aggr_with_bias.transpose(0,1)
        
        return aggr_with_bias


    def _get_heterograph_from_aggregated_adj(self,adj:torch.Tensor,src_hg:dgl.DGLHeteroGraph)->dgl.DGLHeteroGraph:
        r"""
        The function self._multiplex_relation_aggregation returns an adjacency matrix.
        However, we need convert it to DGLHeterograph to process.
        This function convert a graph representation from adjacency matrix to dgl heterograph

        Parameters
        ----------
        adj: torch.Tensor
            A graph representated in adjacency matrix.
        src_hg: dgl.DGLHeteroGraph
            Source hetegraph.
        Returns
        -------
        hg:torch.Tensor
            The graph strtucture is consistent with src_hg.
            The attributes are consistent with adj.
        """
        heterograph_data = dict()
        for src_ntype,src_index in self.node_index_map.items():
            for dst_ntype,dst_index in self.node_index_map.items():
                # slice matrix
                row_start = src_index
                row_end = src_index + self.node_num_map[src_ntype]
                col_start = dst_index
                col_end = dst_index + self.node_num_map[dst_ntype]

                sub_adj = adj[row_start:row_end,col_start:col_end]
                sub_adj_numpy = sub_adj.cpu().detach().numpy()
                idx = sub_adj_numpy.nonzero() # (row, col)

                idx_t = sub_adj.nonzero().T

                if(len(idx[0]) != 0):
                    data = sub_adj[idx]
                    coo_adj = torch.sparse_coo_tensor(idx_t, data, sub_adj.shape).coalesce()
                    coo_adj_indices = coo_adj.indices()


                    heterograph_data[(src_ntype,"aggregated_relation",dst_ntype)] = (coo_adj_indices[0],coo_adj_indices[1])

        hg = dgl.heterograph(heterograph_data)
        for ntype in src_hg.ntypes:
            for ndata_key in src_hg.nodes[ntype].data.keys():
                hg.nodes[ntype].data[ndata_key] = src_hg.nodes[ntype].data[ndata_key]
        return hg
    def _get_graph_from_aggregated_adj(self,adj:torch.Tensor,src_hg:dgl.DGLHeteroGraph)->dgl.DGLHeteroGraph:
        r"""
        The function self._multiplex_relation_aggregation returns an adjacency matrix.
        However, we need convert it to DGLHeterograph to process.
        This function convert a graph representation from adjacency matrix to dgl graph

        Parameters
        ----------
        adj: torch.Tensor
            A graph representated in adjacency matrix.
        src_hg: dgl.DGLHeteroGraph
            Source hetegraph.
        Returns
        -------
        hg:torch.Tensor
            The graph strtucture is consistent with src_hg.
            The attributes are consistent with adj.
        """
        adj_numpy = adj.cpu().detach().numpy()
        idx = adj_numpy.nonzero() # (row, col)
        data = adj[idx]

        idx_t = adj.nonzero().T


        coo_adj = torch.sparse_coo_tensor(idx_t, data, adj.shape).coalesce()
        coo_adj_indices = coo_adj.indices()
        graph_data = (coo_adj_indices[0],coo_adj_indices[1])

        g = dgl.graph(graph_data)

        return g

    def _dict_feature_to_adjacency_feature(self,feature:dict)->torch.Tensor:
        r"""
        In order to process features in GCN(instead of HGCN) layer,we need to convert the feature
        format from dict{str:torch.Tensor} to torch.Tensor.
        
        Considering that after self._multiplex_relation_aggregation, there is only one relation 
        in the graph. So that we can represent the graph in one adjacency matrix in which different
        types of nodes are arranged in a specific order so that we can easily recover it to dict. 

        Parameters
        ----------
        adj: dict[str,torch.Tensor]
            Input features.
        Returns
        -------
        adj_feature:torch.Tensor
            Output feature.
        """
        adj_feature = torch.Tensor(0,self.in_dim).to(self.device)
        for ntype,index in self.node_index_map.items():
            assert(adj_feature.shape[0] == index,"Feature shape not match.")
            adj_feature = torch.cat((adj_feature,feature[ntype]),0)
        return adj_feature
    
    def _adjacency_embedings_to_dict_embedings(self,adj_embedings:torch.Tensor)->dict:
        r"""
        The GCN(instead of HGCN) layer returns a type of torch.Tensor but we prefer to process it 
        by dict[str:torch.Tensor].

        The list node_num_map saves the node type order and index in adjacency.
        By referring it we can convert embedings from torch.Tensor to dict[str:torch.Tensor].

        Parameters
        ----------
        adj_embedings: torch.Tensor 
            Input features.
        Returns
        -------
        dict_embedings:dict[str,torch.Tensor]
            Output feature.
        """
        dict_embedings = dict()
        for ntype,index in self.node_index_map.items():
            row_start = index
            row_end = index + self.node_num_map[ntype]
            sub_adj_embedings = adj_embedings[row_start:row_end,:]
            dict_embedings[ntype]= sub_adj_embedings
        return dict_embedings

            
    
    def _forward_hgcn(self,src_hg:dgl.DGLHeteroGraph,feature:dict):
        r"""
        Parameters
        ----------
        src_hg: dgl.HeteroGraph
            Input graph
        feature: dict[str:torch.Tensor]
            Input feature
        Returns
        -------
        feature:dict[str:torch.Tensor]
            output feature
        """
        aggr_adj = self._multiplex_relation_aggregation(src_hg)

        hg = self._get_heterograph_from_aggregated_adj(aggr_adj,src_hg)


        layer_output_accumulate = {ntype:torch.zeros(value.shape[0],self.out_dim).to(self.device) for ntype,value in feature.items()}
        layer_output = feature
        layer_count = 0
        for index,layer in enumerate(self.layers):

            layer_output = layer(hg,layer_output)

            if {ntype:value.shape for ntype,value in layer_output.items()} == {ntype:value.shape for ntype,value in layer_output_accumulate.items()}:
                layer_output_accumulate = {ntype:layer_output_accumulate[ntype] + layer_output[ntype] for ntype,value in layer_output_accumulate.items()}
                layer_count += 1

        layer_output_average = {ntype:value / layer_count for ntype,value in layer_output_accumulate.items()}
        return layer_output_average
        
    def _forward_gcn(self,src_hg:dgl.DGLHeteroGraph,feature:dict):
        r"""
        Parameters
        ----------
        src_hg: dgl.HeteroGraph
            Input graph
        feature: dict[str:torch.Tensor]
            Input feature
        Returns
        -------
        feature:dict[str:torch.Tensor]
            output feature
        """
        aggr_adj = self._multiplex_relation_aggregation(src_hg)

        g = self._get_graph_from_aggregated_adj(aggr_adj,src_hg)
        adj_feature = self._dict_feature_to_adjacency_feature(feature)
        layer_output_accumulate = torch.Tensor(adj_feature.shape[0],self.out_dim).to(self.device)
        layer_output = adj_feature
        layer_count = 0
        for index,layer in enumerate(self.layers):
            layer_output = layer(g,layer_output)

            layer_output_accumulate += layer_output
            layer_count += 1
        layer_output_average = layer_output_accumulate / layer_count
        layer_output_average_dict = self._adjacency_embedings_to_dict_embedings(layer_output_average)
        return layer_output_average_dict
    def forward(self,src_hg:dgl.DGLHeteroGraph,feature):
        r"""
        Parameters
        ----------
        src_hg: dgl.HeteroGraph
            Input graph
        feature: dict[str:torch.Tensor]
            Input feature
        Returns
        -------
        feature:dict[str:torch.Tensor]
            output feature
        """
        if self.use_hgcn:
            return self._forward_hgcn(src_hg,feature)
        else:
            return self._forward_gcn(src_hg,feature)
        
    


        
        
