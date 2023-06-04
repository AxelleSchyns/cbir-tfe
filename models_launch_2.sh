
#!/usr/bin/env bash

python database/models.py --model resnet  --weights weights_folder/111_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15 --generalise 1 --batch_size 256 --parallel


python database/models.py --model resnet  --weights weights_folder/112_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15 --generalise 2 --batch_size 256 --parallel


python database/models.py --model resnet  --weights weights_folder/113_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15  --batch_size 128


python database/models.py --model resnet  --weights weights_folder/114_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15 --batch_size 32 --parallel