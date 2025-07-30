# Outlier Interpretation
[Uploading AAAI.pdf…]()


This repository contains the source code for the paper **Density-Aware Hypergraph Networks for Industrial Outlier Interpretation**.

Note that this task is also referred to as outlier explanation, outlier aspect mining/discovering, outlier property detection, and outlier description.

### Structure
`data_od_evaluation`: Ground-truth outlier interpretation annotations of real-world datasets  
`data`: real-world datasets in csv format, the last column is label indicating each line is an outlier or a inlier  
`model_xx`: folders of ATON and its contenders, the competitors are introduced in Section 5.1.2  
`config.py`: configuration and default hyper-parameters  
`main.py` main script to run the experiments



### How to use?
##### 1. For ATON and competitor COIN, SHAP, and LIME
1. modify variant `algorithm_name` in `main.py` (support algorithm: `aton`, `coin`, `shap`, `lime`  in lowercase)
2. use `python main.py --path data/ --runs 10 `
3. the results can be found in `record/[algorithm_name]/` folder  

##### 2. For SPELT and competitor COIN 
1. modify variant `algorithm_name` in `main.py` to `SPELT` or `coin`  
2. use `python main.py --path data/ --w2s_ratio auto --runs 10` to run SPELT  
   use `python main.py --path data/ --w2s_ratio pn --runs 10` to run COIN  


### Requirements
main packages of this project  
```
torch==1.3.0
numpy==1.15.0
pandas==0.25.2
scikit-learn==0.23.1
pyod==0.8.2
tqdm==4.48.2
prettytable==0.7.2
shap==0.35.0
lime==0.2.0.1
alibi==0.5.5
```



### Ground-truth annotations

Please also find the Ground-truth outlier interpretation annotations in folder `data_od_evaluation`.   
*We expect these annotations can foster further possible reasearchs on this new practical probelm.*  

**How to generate the ground-truth annotations:**
>  We employ three different kinds of representative outlier detection methods (i.e., ECOD \cite{li2022ecod}, COPOD \cite{li2020copod}, iForest \cite{liu2012isolation}, HBOS \cite{goldstein2012hbos}, ROD \cite{almardeny2020novel}, and MCD \cite{hardin2004outlier}) to evaluate outlying degree of real outliers given every possible subspace. A good explanation for an outlier should be a high-contrast subspace that the outlier explicitly demonstrates its outlierness, and outlier detectors can easily and certainly predict it as an outlier in this subspace. Therefore, the ground-truth interpretation for each outlier is defined as the subspace that the outlier obtains the highest outlier score among all the possible subspaces.



### References
- datasets are from ODDS, an outlier detection datasets library (http://odds.cs.stonybrook.edu/), and kaggle platform (https://www.kaggle.com/)
