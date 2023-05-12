
#!/usr/bin/env bash
python database/models.py --model resnet  --weights weights_folder/88_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15 --loss contrastive --non_contrastive --batch_size 128 --gpu 1


python database/models.py --model resnet  --weights weights_folder/87_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15 --loss contrastive --batch_size 128 --gpu 1

