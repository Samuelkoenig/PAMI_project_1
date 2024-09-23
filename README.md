# Automatic and Transparent Inspection Model

## Prerequisites

-   Python (https://www.python.org/downloads)
-   Jupyter Notebooks (https://jupyter.org/install)

## Getting Started

1. Clone repository

    ```bash
    git clone https://github.com/Samuelkoenig/PAMI_project_1.git
    ```

2. Navigate into the repository

    ```bash
    cd PAMI_project_1
    ```

3. Install Python dependencies

    ```bash
    pip install -r requirements.txt
    ```

4. Cretae local data folder

   Create a data folder containing the dacl10k data.

## Components

| File                                                         | Purpose                                                                            |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| image_processing.py                                          | Python file to generate the dataset from the raw input images and annotations.     |
| feature_selection/feature_selection.ipynb                    | Jupyter Notebook to perform the feature selection.                                 |
| parameter_tuning/parameter_tuning.py                         | Python file to perform the parameter tuning using binary classifications.          |
| parameter_tuning/cross_classification.py                     | Python file to perform the parameter tuning using a combined classification.       |
| multilabel_classification/multilabel_classification.ipynb    | Jupyter Notebook to train a multilabel classifier (not analyzed in final report).  |
