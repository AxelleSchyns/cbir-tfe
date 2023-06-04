
#!/usr/bin/env bash

#printf "50 resnet \n"
#python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/50_resnet --rewrite --generalise 3

#printf "50 resnet: all \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/50_resnet --measure all --generalise 3

#printf "50 resnet: weighted \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/50_resnet --measure weighted --generalise 3


printf "82 auto \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor auto --weights weights_folder/82_auto --rewrite --num_features 3072
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor auto --weights weights_folder/82_auto --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/82_auto --num_features 3072


printf "22 resnet18\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet18 --weights weights_folder/102_resnet18 --rewrite --num_features 25088
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor resnet18 --weights weights_folder/102_resnet18 --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/22_resnet18 --num_features 25088


printf "29 vae\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vae --weights weights_folder/103_vae --rewrite --num_features 20
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor vae --weights weights_folder/103_vae --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/29_vae --num_features 20


printf "50 resnet\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/50_resnet --rewrite --generalise 3
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor resnet --weights weights_folder/50_resnet --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/40_resnet --generalise 3


printf "47 byol\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor byol --weights weights_folder/109_byol --rewrite --num_features 256 
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor byol --weights weights_folder/109_byol --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/47_byol --num_features 256



#printf "116 byol + test\n"
#python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor byol --weights weights_folder/116_byol --rewrite --num_features 256 --generalise 2

#printf "116 byol: all \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor byol --weights weights_folder/116_byol --measure all --num_features 256 -

#printf "116 byol: weighted \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor byol --weights weights_folder/116_byol --measure weighted --num_features 256 --generalise 2




#printf "104 resnet training \n"
#python database/models.py --model resnet  --weights weights_folder/104_resnet --training_data /home/labarvr4090/Documents/Axelle/cytomine/Data/train --scheduler exponential  --num_epochs 15 --generalise 3 --batch_size 256 --parallel

#printf "104 resnet + test\n"
#python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/104_resnet --rewrite --generalise 3

#printf "104 resnet: all \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/104_resnet --measure all --generalise 3

#printf "104 resnet: weighted \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/104_resnet --measure weighted --generalise 3