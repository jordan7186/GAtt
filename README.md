# Source code for GAtt

This repository is the official implementation of [Revisiting Attention Weights as Interpretations of Message-Passing Neural Networks](https://arxiv.org/abs/2030.12345](https://arxiv.org/abs/2406.04612)). In the paper, we show that GAtt provides a better way to calculate edge attribution scores from attention weights in attention-based GNNs!

## We have a Dockerfile for running locally:

Just clone the repo, and build the docker image by:

```bash
docker build <directory_to_cloned_repo> --tags <your:tags>
```

### If you don't want Docker...

The code is tested in...
- Python 3.10.13
- Pytorch 2.0.1+cu117
- Pytorch Geometric 2.3.1

which should be enough to run the demos.

## Provided in this repo are...
1. Source code for **GAtt**
2. Demos
- Demo on the Cora dataset on how to use the `get_gatt` function
- Demo on the BAShapes (generated from `torch_geometric.datasets.ExplainerDataset`): Visualizations of **GAtt** and comparison to AvgAtt
- Demo on the Infection dataset (generated from the code in [the original authors' repo](https://github.com/m30m/gnn-explainability)): Visualizations of **GAtt** and comparison to AvgAtt

## Results for Infection dataset

This is one of the results in the demo notebooks:

Figure (left to right)
- Ground truth explanation (blue edges) for the target node (orange node)
- Edge attribution from **GAtt**
- Edge attribution from AvgAtt (averaging over the layers)

<p float="left">
  <img src="/Figures/Infection_3L_ground_truth.png" width="250" />
  <img src="/Figures/Infection_3L_GAtt.png" width="250" /> 
  <img src="/Figures/Infection_3L_AvgAtt.png" width="250" />
</p>

The figures show that the edge attribution scores in GAtt is more aligned with the ground truth explanation edges compared to just averaging over the GAT layers.


## Note on batch computations of GAtt
This will be updated soon...
