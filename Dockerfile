FROM python:3.10

RUN mkdir -p /workspace

WORKDIR /workspace

RUN pip install torch==2.0.1 torch_geometric==2.3.1 matplotlib ipykernel jupyter
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html