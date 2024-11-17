# FDML
## Paper
Conditional generation model with dual-perspective feature fusion representation for multi-label classification
## Requirtments
python 3.7 torch 1.13 scikit-learn 1.3.0 tensorboard 2.11.1 scipy 1.10.1
## Demo: Corel16k001
Train: input parameters according to the file 'params/Corel16k001.json'
       model:train
       run main.py
Test: input checkpoint_path according to the file 'model/model_Corel16k001/lr-0.001_lr-decay_0.03_lr-times_4.0/FDML-1890'
      model: test
      run main.py
