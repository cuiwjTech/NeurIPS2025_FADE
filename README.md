# Neural Fractional Attention Differential Equations
This repository contains the code for our NeurIPS 2025 accepted paper, Neural Fractional Attention Differential Equations

## Reproducing Results
To run our code, go to the /src folder.


```bash
python run_GNN_frac_all.py 
--dataset  Cora, Citeseer, Pubmed, CoauthorCS, CoauthorPhy, Computers, Photo
--function laplacian/ transformer
--block constant_frac/ att_frac
--method predictor/ predictor_corrector
--alpha_ode  between (0,1] the value of beta in the paper
--time   integration time 
--step_size  

FOR EXAMPLE:

run_GNN_frac_all.py --dataset Cora --function laplacian --block att_frac --cuda 1 --method predictor --epoch 400 --seed 123 --runtime 10 --decay 0.01 --dropout 0.2 --hidden_dim 256 --input_dropout 0.6 --alpha_ode 0.85 --time 40 --step_size 1.0 --lr 0.01

```

## Reference 

Our code is developed based on the following repo:  

The graph neural ODE model is based on the ICLR 2024 [FROND](https://github.com/zknus/ICLR2024-FROND) framework.  


## Citation 

If you find our work useful, please cite us as follows:
```
@INPROCEEDINGS{kangneural,
  title={Neural Fractional Attention Differential Equations},
  author={Qiyu Kang and Wenjun Cui and Xuhao Li and Yuxin Ma and Xueyang Fu and Wee Peng Tay and Yidong Li and Zhengjun Zha},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
