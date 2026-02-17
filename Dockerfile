FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace

RUN pip install --no-cache \
    --extra-index-url https://pypi.nvidia.com \
    cudf-cu12==24.4.* \
    cuml-cu12==24.4.* \
    cupy-cuda12x \
    xgboost \
    torchmetrics \
    scikit-learn \
    pandas

CMD ["bash"]