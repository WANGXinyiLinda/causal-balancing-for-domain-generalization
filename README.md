# Causal Balancing for Domain Generalization

This is the official source code of the paper [Causal Balancing for Domain Generalization](https://arxiv.org/abs/2206.05263). 
We propose a balanced mini-batch sampling method, which can be incorporated into other domain generalization algorithms.
A large portion of the code is modified from [DomainBed](https://github.com/facebookresearch/DomainBed), which is a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434). 

Our method has two steps: 
1. *latent covariate learning* that learns an VAE on the observed data distribution to calculate propensity score; 
2. *Balacing score matching* that match examples with the closest propensity score to create balanced mini-batches for training base algorithm.

## 0. Setup the repository

Install the required packages with python >= 3.7:
```sh
python -m pip install -r requirements.txt
```

Download the datasets:
```sh
python -m domainbed.scripts.download \
       --data_dir $DATA_DIR
```

-------

## 1. Extract base algorithm hyperparameters

In our paper, we extract optimal hyperparameters of the base algorithms from the full DomainBed sweep log posted [here](https://drive.google.com/file/d/16VFQWTble6-nB5AdXBtQpQFwjEC7CChM/view?usp=sharing). 
To extract the optimal base algorithm hyperparameters, use the following command:
```sh
python -m domainbed.scripts.collect_results_detailed\
       --input_dir $FULL_SWEEP_LOG
```
Set `FULL_SWEEP_LOG` to the path of decompressed sweep log. Selected hyperparameters will be stored at `$FULL_SWEEP_LOG/chosen_records.json`.

To verify the correctness of the obtained optimal hyperparameters with 
**train domain validation**, you can create a sweep using the following command:
```sh
python -m domainbed.scripts.sweep_selected launch\
       --data_dir $DATA_DIR\
       --selected_records $FULL_SWEEP_LOG/chosen_records.json\
       --output_dir $SWEEP_OUT\
       --command_launcher local\
       --algorithms $CLF_ALG\
       --datasets $DATASET\
       --use_gpu\
       --single_test_envs 
```
Set `CLF_ALG` to the desired base algorithm(s), `DATASET` to the desired dataset(s) and `SWEEP_OUT` to the desired output directory.

-------

## 2. Replicate our experiments

To create a sweep with the optimal hyperparameters and balanced mini-batches, use the following command:
```sh
python -m domainbed.scripts.sweep_selected launch\
       --data_dir $DATA_DIR\
       --selected_records $FULL_SWEEP_LOG/chosen_records.json\
       --output_dir $SWEEP_OUT\
       --command_launcher local\
       --algorithms $CLF_ALG\
       --match_algorithms CVAE \
       --datasets $DATASET\
       --num_cf $NUM_CF\
       --use_gpu \
       --single_test_envs \
```
Set `CLF_ALG` to the desired base algorithm(s), `DATASET` to the desired dataset(s), `SWEEP_OUT` to the desired output directory and `NUM_CF` to the appropriate number of matched examples.

The above script will first train a CVAE to learn the latent covariate and compute the balancing score for each train example. Then balanced mini-batches will be constructed to train base algorithms with the extracted optimal hyperparameters. Each experiment will be repeated for 3 times.

------

## 3. View sweep results

To view your sweep results, use the following command:
```sh
python -m domainbed.scripts.collect_results\
       --input_dir $SWEEP_OUT
```
We report **train domain validation** results in our paper.

------

## 4. Add new datasets

You can add your own dataset by implementing a subclass of `MultipleDomainDataset` in `domainbed/datasets.py` to load your dataset. You will be able to use your dataset by providing the dataset class name to the `--dataset` argument. The core algorithm of causal matching is in `domainbed/match`. You can also transfer the algorithm to your own codebase.

------

## Citation

```
@inproceedings{
wang2023causal,
title={Causal Balancing for Domain Generalization},
author={Xinyi Wang and Michael Saxon and Jiachen Li and Hongyang Zhang and Kun Zhang and William Yang Wang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=F91SROvVJ_6}
}
```