<div align="center">
    <p>
    <h1>
    GRAPE - Implementation
    </h1>
</div>



## Description
Learning Graph-based Patch Representations for Identifying and Assessing Silent Vulnerability Fixes
![alt text](image/overview.png)
## Requirement
Our code is based on Python3 (>= 3.8). There are a few dependencies to run the code. The major libraries are listed as follows:

- torch (==2.0.1)
- pyg (==2.4.0)
- torch_scatter (==2.1.2+pt20cu118)
- torch-cluster (==1.6.3+pt20cu118)
- torch-sparse (==0.6.18+pt20cu118)
- numpy (==1.24.3)
- pandas (==2.0.3)
- tqdm (==4.65.0)

## Dataset
The dataset is in the `~/GRAPE/data/` folder
- commit_label.csv for vulnerability fix identification task
- commit_cwe.csv for vulnerability types classification task
- commit_cve.csv for vulnerability severity classification task 
- dataset.csv : vulnerability fix dataset
## How-to-Run
### Preprocess
1. Using `Joern` to generate CPGs
In `~/GRAPE/`, run the command:
```shell
sudo python3 generate_cpg.py
```
2. Merging CPGs to MCPG
```shell
python3 merge_cpg.py
```

### Train 
1. Processing MCPG into vectors
```shell
python3 preprocess.py
```
2. Training the model
```shell
python3 train.py
```
## 5. Team
The GRAPE package was developed by Institute of Software Engineering, Southeast University(ISEU).
