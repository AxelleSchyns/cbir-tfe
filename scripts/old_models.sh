#!/usr/bin/env bash

conda activate tfe

python database/models.py --model densenet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --num_epochs 15 --weights weights_folder/2_densenet

python database/models.py --model resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --num_epochs 15 --weights weights_folder/3_resnet

python database/models.py --model deit --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --num_epochs 15 --weights weights_folder/4_deit

python database/models.py --model knet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --num_epochs 15 --weights weights_folder/5_kimianet

python database/models.py --model knet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --num_epochs 15 --weights weights_folder/35_kimianet --scheduler exponential --dr_model 




