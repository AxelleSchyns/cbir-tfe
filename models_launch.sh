
#!/usr/bin/env bash


printf "103 vae"
python database/models.py --model resnet  --weights weights_folder/test_epoch --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 1 --batch_size 64 --loss infonce --non_contrastive

printf "82 auto \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor auto --weights weights_folder/82_auto --rewrite --num_features 3072
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor auto --weights weights_folder/82_auto --measure stat --num_features 3072
#printf "weighted\n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor auto --weights weights_folder/82_auto --measure weighted --num_features 3072
#printf "all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor auto --weights weights_folder/82_auto --measure all --num_features 3072

