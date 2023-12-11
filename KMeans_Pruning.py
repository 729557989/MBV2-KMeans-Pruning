import copy
import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from MobileNetV2 import MobileNetV2
from helpers import visualize_weights
"""
NOTEs:
    - 最好是能复现一个随着pruning 然后acc下降的过程
    - First 3 conv layers are not pruned. The last conv layer is the additional pointwise conv. The last linear layer follows.
    - ALl conv layers is 52, last linear layer is 1, so total of 53 layers.
    
Pruning Instructions:
    1. The first regular CNN + first BottleNeck (one with only a depthwise followed by pointwise CNN) are untounched.
    
    2. For each subsequent BottleNeck's layers:
        - pointwise A: Adjust output filters based on depthwise B's pruning indices.
            [64 (filters), 32 (in dim), 1, 1] -> [48 (filters), 32 (in dim), 1, 1]
            
        - dehpthwise B: Do KMeans clustering, prune output filters
            [64 (filters and in dim), 1, 3, 3] -> [48 (filters and in dim), 1, 3, 3]
            
        - pointwise C: Adjust input filters based on depthwise B's pruning indices.
            [32 (filters), 64 (in dim), 1, 1] -> [32 (filters), 48 (in dim), 1, 1]
"""
class KMeansPruning:
    def __init__(self, model, sparsity=0.8) -> None:
        self.previous_model = model
        self.model = None
        self.sparsity = sparsity
    
    # Determine the best k value using The Silhouette Method
    def _get_optimal_k(self, k_range_upperbound, cluster_datapoints):
        sil = []
        for k in range(2, k_range_upperbound-1):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(cluster_datapoints)
            labels = kmeans.labels_
            sil.append((k, silhouette_score(cluster_datapoints, labels, metric = 'euclidean')))

        optimal_k = max(sil, key = lambda x: x[1])[0]
        
        return optimal_k
    
    def pruned_dims(self):
        assert self.model != None, "You need to PRUNE the model by running '.get_pruned_model' first!"
        print("\n")
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                print(m)
                # print(m, end="  ->  ")
                # print(m.weight.data.shape)
        print("\n")
    
    def _indices_to_keep(self, conv_layer):
        # Initialize a list to store the indices of filters to keep
        to_keep = []

        # Determine the number of filters in the convolutional layer
        num_filters = conv_layer.weight.data.size(0)
        
        # The total number of filters to prune from the convolutional layer
        total_to_prune = int((1-self.sparsity) * num_filters)

        # Reshape the weight data of the convolutional layer for clustering
        cluster_datapoints = conv_layer.weight.data.reshape(num_filters, -1)

        # Determine the optimal number of clusters using the elbow method
        optimal_k = self._get_optimal_k(num_filters, cluster_datapoints)

        # Perform KMeans clustering on the reshaped weight data
        kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init="auto").fit(cluster_datapoints)
        labels = kmeans.labels_

        """
        1. Organize filters into clusters and compute the L1 norm for each filter
        2. Each cluster is represented as a dictionary entry
        3. The key is the cluster label, and the value is a list of tuples.
        4. Each tuple contains the index of a filter in that cluster and its L1 norm
        """
        cluster_groups = {i: [] for i in range(optimal_k)}
        for i, label in enumerate(labels):
            index_and_L1 = (i, torch.sum(torch.abs(conv_layer.weight.data[i])))
            cluster_groups[label].append(index_and_L1)
        
        # For each cluster, sort filters by their L1 norm and select a subset to keep
        for label, weights in cluster_groups.items():
            # Calculate number of filters to prune per cluster
            proportion = len(weights) / num_filters
            cluster_to_prune = int(proportion * total_to_prune)
            # Sort weights in ascending order based on L1 norm
            sorted_weights = sorted(weights, key=lambda x: x[1])
            # Extract sorted indices of filters
            sorted_indices = [index for (index, L1) in sorted_weights]
            # Extend the 'to_keep' list with the indices of filters to retain
            to_keep.extend(sorted_indices[cluster_to_prune:])

        # Return the sorted list of indices of filters to keep
        return sorted(to_keep)

    @property
    def get_pruned_model(self):
        start_time = time.time()     
        
        # Create a copy of the model to prune
        self.model = copy.deepcopy(self.previous_model)
        # Get the original model's conv layers for printing out dimensions before pruning
        primitive_all_convs = [m for m in self.previous_model.modules() if isinstance(m, nn.Conv2d)][3:] # Skip the first 3 conv layers
        # Get all the conv layers of the model to prune
        all_convs = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)][3:] # Skip the first 3 conv layers
        # Get all the bn layers of the model to prune
        all_bns = [m for m in self.model.modules() if isinstance(m, nn.BatchNorm2d)][3:] # Skip the first 3 conv layers
        assert len(all_convs) == len(all_bns)
        
        # Iterate through all the conv layers from BottleNecks and prune them
        for idx, _ in enumerate(all_convs):
            curr_conv = all_convs[idx]
            if curr_conv.groups > 1: # Reached a depthwise conv
                pre_conv = all_convs[idx-1]
                next_conv = all_convs[idx + 1] # currently we dont need to check idx == len(all_convs) - 1 beacuse we are not pruning the pointwise conv layer
                prev_bn = all_bns[idx-1]
                curr_bn = all_bns[idx]
                
                # Perform KMeans clustering to get indices for filters to prune
                to_keep = self._indices_to_keep(curr_conv)
                
                with torch.no_grad():
                    # 1. Prune the previous pointwise conv layer's output filters + bn layer's weights
                    pre_conv.out_channels = len(to_keep)
                    pre_conv.weight.set_(torch.index_select(pre_conv.weight.detach(), dim=0, index=torch.tensor(to_keep)))
                    
                    prev_bn.num_features = len(to_keep)
                    prev_bn.weight.set_(torch.index_select(prev_bn.weight.detach(), dim=0, index=torch.tensor(to_keep)))
                    prev_bn.bias.set_(torch.index_select(prev_bn.bias.detach(), dim=0, index=torch.tensor(to_keep)))
                    prev_bn.running_mean.set_(torch.index_select(prev_bn.running_mean.detach(), dim=0, index=torch.tensor(to_keep)))
                    prev_bn.running_var.set_(torch.index_select(prev_bn.running_var.detach(), dim=0, index=torch.tensor(to_keep)))
                    
                    # 2. Prune the current depthwise conv layer's output filters + bn layer's weights
                    curr_conv.out_channels = len(to_keep)
                    curr_conv.groups = len(to_keep)
                    curr_conv.weight.set_(torch.index_select(curr_conv.weight.detach(), dim=0, index=torch.tensor(to_keep)))
                    
                    curr_bn.num_features = len(to_keep)
                    curr_bn.weight.set_(torch.index_select(curr_bn.weight.detach(), dim=0, index=torch.tensor(to_keep)))
                    curr_bn.bias.set_(torch.index_select(curr_bn.bias.detach(), dim=0, index=torch.tensor(to_keep)))
                    curr_bn.running_mean.set_(torch.index_select(curr_bn.running_mean.detach(), dim=0, index=torch.tensor(to_keep)))
                    curr_bn.running_var.set_(torch.index_select(curr_bn.running_var.detach(), dim=0, index=torch.tensor(to_keep)))
                    
                    # 3. Prune the next pointwise conv layer's input filters
                    next_conv.in_channels = len(to_keep)
                    next_conv.weight.set_(torch.index_select(next_conv.weight.detach(), dim=1, index=torch.tensor(to_keep)))
            
                print(f"Prev_dim: {primitive_all_convs[idx-1].weight.data.shape} -> Curr_dim: {pre_conv.weight.data.shape}")
                print(f"Prev_dim: {primitive_all_convs[idx].weight.data.shape} -> Curr_dim: {curr_conv.weight.data.shape}")
                print(f"Prev_dim: {primitive_all_convs[idx+1].weight.data.shape} -> Curr_dim: {next_conv.weight.data.shape}")
        
        # Prune the last pointwise conv layer (w/ KMeans Clustering) and adjust the last linear layer.
        last_conv_layer = all_convs[-1]
        last_bn_layer = all_bns[-1]
        to_keep = self._indices_to_keep(last_conv_layer)
        
        with torch.no_grad():
            last_conv_layer.out_channels = len(to_keep)
            last_conv_layer.weight.set_(torch.index_select(last_conv_layer.weight.detach(), dim=0, index=torch.tensor(to_keep)))
            
            last_bn_layer.num_features = len(to_keep)
            last_bn_layer.weight.set_(torch.index_select(last_bn_layer.weight.detach(), dim=0, index=torch.tensor(to_keep)))
            last_bn_layer.bias.set_(torch.index_select(last_bn_layer.bias.detach(), dim=0, index=torch.tensor(to_keep)))
            last_bn_layer.running_mean.set_(torch.index_select(last_bn_layer.running_mean.detach(), dim=0, index=torch.tensor(to_keep)))
            last_bn_layer.running_var.set_(torch.index_select(last_bn_layer.running_var.detach(), dim=0, index=torch.tensor(to_keep)))
            
            self.model.classifier.in_features = len(to_keep)
            self.model.classifier.weight.set_(torch.index_select(self.model.classifier.weight.detach(), dim=1, index=torch.tensor(to_keep)))
        
        print(f"Prev_dim: {last_conv_layer.weight.data.shape} -> Curr_dim: {primitive_all_convs[-1].weight.data.shape}")
        print(f"Prev_dim: {self.previous_model.classifier.weight.data.shape} -> Curr_dim: {self.model.classifier.weight.data.shape}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n Pruning took {elapsed_time} seconds to complete. \n")
        
        return self.model


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"
    
    model = MobileNetV2()
    model.load_state_dict(torch.load(f"models\model_76_epoch.pt", map_location=device))
    model.eval()
    
    # visualize_weights(model)
    
    pruner = KMeansPruning(model, sparsity=0.8)
    model = pruner.get_pruned_model
    pruner.pruned_dims()
    
    model.profile_test()
