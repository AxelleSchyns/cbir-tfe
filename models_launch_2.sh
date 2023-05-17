
#!/usr/bin/env bash

python database/models.py --model resnet  --weights weights_folder/90_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15 --loss infonce --batch_size 128 --gpu 1