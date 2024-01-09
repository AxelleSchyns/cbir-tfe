
#!/usr/bin/env bash

python database/models.py --num_features 128 --batch_size 128 --model deit --training_data '/home/labarvr4090/Documents/Axelle/cytomine/Data/train' --num_epochs 50 --loss softmax --parallel 


python database/models.py --num_features 128 --batch_size 128 --model deit --training_data '/home/labarvr4090/Documents/Axelle/cytomine/Data/train' --num_epochs 50 --loss softmax --parallel --i_sampling False --remove_val