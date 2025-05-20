# Code implementation of our proposed CPF (<u>C</u>oarsening-based <u>P</u>artition-wise <u>F</u>iltering)

## Environmental Details
Ubunto-22.04  
CUDA-11.8  
python == 3.10  
torch == 2.4.0  
torchvision == 0.19.1  
torchaudio == 2.4.1  
torch-geometric == 2.6.0  
pygsp == 0.5.1  
pyg_lib == 0.4.0  
torch_cluster == 1.6.3  
torch_scatter == 2.1.2  
torch_sparse == 0.6.18  
torch_spline_conv == 1.2.2  
sortedcontainers == 2.4.0  
fast-pytorch-kmeans == 0.2.2  

## Usages

#### *Basic Executions*

To train or test the model, use the following unified command:

`python train.py --dataset [DATASET] --r_train [TRAINING_RATIO] --r_val [VALIDATION_RATIO] --lr [LEARNING_RATE] --prop_g_lr [LEARNING_RATE_G] --prop_f_lr [LEARNING_RATE_P] --weight_decay [WEIGHT_DECAY] --prop_g_wd [WD_G] --prop_f_wd [WD_F] --coarsening_ratio [COARSENING_RATIO] --gpu`

#### *Argument Descriptions*

- `--lr`, `--prop_g_lr`, and `--prop_f_lr`: Learning rates for the MLP-based feature transformation, structure-aware filtering, and feature-aware filtering, respectively.
- `--weight_decay`, `--prop_g_wd`, and `--prop_f_wd`: Weight decay for the corresponding components above.
- `--dataset`: Specifies the dataset to use; handled via `dataset.py`. Dataset structure follows the format from [this repository](https://github.com/CUAI/Non-Homophily-Benchmarks) for consistency and compatibility with prior works.
- `--coarsening_ratio`: Sets the coarsening ratio $r$ used in the paper. Several built-in graph coarsening algorithms are available in the `graph_partition` folder. To add custom coarsening methods, ensure they return a *Partition Matrix* and integrate them to `utils.py`.
- `--gpu`: Optional flag to enable GPU acceleration.

#### *Simplified Execution*

For convenience, you can store optimized hyperparameters in `PARAMETERS.py` and run predefined configurations like:

**Roman-empire:**
`python train.py --dataset roman-empire --finetuned --gpu`

**Amazon-ratings**
`python train.py --dataset amazon-ratings --finetuned --gpu`

**Cora:**
`python train.py --dataset cora --finetuned --gpu`

**Citeseer:**
`python train.py --dataset citeseer --finetuned --gpu`

**Pubmed:**
`python train.py --dataset pubmed --finetuned --gpu`

## Notes
- The CPF method is encapsulated in the `CPFGNN` class, aligning with its role as a Graph Neural Network model.

- Slight result variance may occur due to system-dependent randomness. Running multiple trials can help reduce these fluctuations.