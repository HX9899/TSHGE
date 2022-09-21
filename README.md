# TSHGE <br>
# Environment <br>
python 3.7.11 <br>
pytorch 1.7.1 <br>
# Dataset <br>
There are three datasets: YAGO, WIKI and ICEWS14. Each data folder has 'train.txt', 'valid.txt', 'test.txt'. <br> 
# Run the experiments <br>
1. cd ./TSHGE/train <br>
2. Subject entity prediction: python train.py --pred sub
   Object entity prediction: python train.py --pred obj
   Relation prediction: python train.py --pred rel
3. python --dataset YAGO ----lr-conv 0.001 ----time-interval 1 ----n-epochs-conv 50 ----batch-size-conv 50 --pred sub --valid-epoch 5 --count 8 (you can setting parameters this way) <br>
   
