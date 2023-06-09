
#!/usr/bin/env bash

printf "3 efffnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor effnet --weights weights_folder/15_effnet --rewrite 
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet  --weights weights_folder/15_effnet --results_dir results/3_effnet/all_classes 


printf " 4 kimianet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor knet --weights weights_folder/14_kimianet --rewrite
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet  --weights weights_folder/14_kimianet --results_dir results/4_kimianet/all_classes

printf "5 vision \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vision --weights weights_folder/29_vision --rewrite
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vision  --weights weights_folder/29_vision --results_dir results/5_vision/all_classes


printf "9 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/86_resnet --rewrite
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet  --weights weights_folder/86_resnet --results_dir results/9_resnet/all_classes


printf "11 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/97_resnet --rewrite --dr_model
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet  --weights weights_folder/97_resnet --results_dir results/11_resnet/all_classes --dr_model


printf "15 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/90_resnet --rewrite 
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet  --weights weights_folder/90_resnet --results_dir results/15_resnet/all_classes


printf "22 resnet18 \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet18 --weights weights_folder/102_resnet18 --rewrite --num_features 25088
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet18  --weights weights_folder/102_resnet18 --results_dir results/22_resnet18/all_classes --num_features 25088


printf "29 vae \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vae --weights weights_folder/103_vae --rewrite --num_features 20
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vae  --weights weights_folder/103_vae --results_dir results/29_vae/all_classes --num_features 20


printf "33 auto \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor auto --weights weights_folder/82_auto --rewrite --num_features 3072
python database/retrieve_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor auto  --weights weights_folder/82_auto --results_dir results/33_auto/all_classes --num_features 3072

