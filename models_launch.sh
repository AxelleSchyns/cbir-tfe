
#!/usr/bin/env bash

#printf "50 resnet \n"
#python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/50_resnet --rewrite --generalise 3

#printf "50 resnet: all \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/50_resnet --measure all --generalise 3

#printf "50 resnet: weighted \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/50_resnet --measure weighted --generalise 3


printf "71 auto \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor auto --weights weights_folder/71_auto --rewrite 

python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor auto --weights weights_folder/71_auto --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/71_auto


printf "1 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/84_resnet --rewrite
python database/retrieve_images.py --path  /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor resnet --weights weights_folder/84_resnet --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/1_resnet

printf "3 effnet\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor effnet --weights weights_folder/15_effnet --rewrite
python database/retrieve_images.py --path  /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor effnet --weights weights_folder/15_effnet --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/3_effnet

printf "5 vision\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vision --weights weights_folder/29_vision --rewrite
python database/retrieve_images.py --path  /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor vision --weights weights_folder/29_vision --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/5_vision

printf "9 resnet\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/86_resnet --rewrite
python database/retrieve_images.py --path  /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor resnet --weights weights_folder/86_resnet --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/9_resnet

printf "11 resnet\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/97_resnet --rewrite --dr_model
python database/retrieve_images.py --path  /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor resnet --weights weights_folder/97_resnet --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/11_resnet --dr_model

printf "15 resnet\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/90_resnet --rewrite 
python database/retrieve_images.py --path  /home/labarvr4090/Documents/Axelle/cytomine/Data/validation/janowczyk1_1/8867_116795971_0_0_250_250.png --extractor resnet --weights weights_folder/90_resnet --results_dir /home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/results/15_resnet
#python database/rec_images.py --path  /home/labarvr4090/Documents/Axelle/cytomine/Data/test/cells_no_aug_0/728755_748504.png --extractor resnet18 --weights weights_folder/102_resnet18  --namefig "rec_102_resnet18" 

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