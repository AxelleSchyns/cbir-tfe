
#!/usr/bin/env bash
python database/models.py --model resnet  --weights weights_folder/91_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15 --batch_size 256 --parallel 

python database/models.py --model resnet  --weights weights_folder/92_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15  --batch_size 256 --parallel --freeze 

python database/models.py --model resnet  --weights weights_folder/93_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15  --batch_size 256 --parallel --scratch