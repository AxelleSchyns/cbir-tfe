#!/usr/bin/env bash

printf "Densenet 10: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor densenet --weights weights_folder/10_densenet --rewrite --dr_model
printf "Densenet 10: random\n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/10_densenet --dr_model
printf "Densenet 10: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/10_densenet --dr_model --measure remove
printf "Densenet 10: all\n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/10_densenet  --dr_model --measure all

printf "Resnet 11: indexing\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/11_resnet --rewrite --dr_model
printf "Resnet 11: random\n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/11_resnet --dr_model
printf "Resnet 11: remove\n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/11_resnet --dr_model --measure remove
printf "Resnet 11: all\n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/11_resnet --dr_model --measure all

printf "Knet 14: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor knet --weights weights_folder/14_kimianet --rewrite
printf "KNet 14: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet --weights weights_folder/14_kimianet
printf "KNet 14: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet --weights weights_folder/14_kimianet --measure remove
printf "KNet 14: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet --weights weights_folder/14_kimianet --measure all

printf "Effnet 15: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor effnet --weights weights_folder/15_effnet --rewrite
printf "Effnet 15: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet --weights weights_folder/15_effnet
printf "Effnet 15: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet --weights weights_folder/15_effnet --measure remove
printf "Effnet 15: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet --weights weights_folder/15_effnet --measure all

printf "Densenet 21: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor densenet --weights weights_folder/21_densenet --rewrite --dr_model
printf "Densenet 21: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/21_densenet --dr_model
printf "Densenet 21: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/21_densenet --measure remove --dr_model
printf "Densenet 21: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/21_densenet --measure all --dr_model

printf "Densenet 22: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor densenet --weights weights_folder/22_densenet --rewrite
printf "Densenet 22: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/22_densenet
printf "Densenet 22: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/22_densenet --measure remove
printf "Densenet 22: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/22_densenet --measure all

printf "Effnet 27: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor effnet --weights weights_folder/27_effnet --rewrite --dr_model
printf "Effnet 27: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet --weights weights_folder/27_effnet --dr_model
printf "Effnet 27: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet --weights weights_folder/27_effnet --dr_model --measure remove
printf "Effnet 27: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet --weights weights_folder/27_effnet --dr_model --measure all

printf "Densenet 31: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor densenet --weights weights_folder/31_densenet --rewrite
printf "Densenet 31: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/31_densenet
printf "Densenet 31: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/31_densenet --measure remove
printf "Densenet 31: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/31_densenet --measure all 


printf "Densenet 37: random \n"
python database/classification_acc.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/37_densenet
printf "Densenet 37: remove \n"
python database/classification_acc.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/37_densenet --measure remove
printf "Densenet 37: all \n"
python database/classification_acc.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/37_densenet --measure all 

printf "Autoencoder 34: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet18 --weights weights_folder/34_autoencoder.pth --rewrite --num_features 25088
printf "Autoencoder 34: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet18 --weights weights_folder/34_autoencoder.pth --num_features 25088
printf "Autoencoder 34: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet18 --weights weights_folder/34_autoencoder.pth --measure remove --num_features 25088
printf "Autoencoder 34: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet18 --weights weights_folder/34_autoencoder.pth --measure all --num_features 25088

printf "Knet 39: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor knet --weights weights_folder/39_kimianet --rewrite
printf "Knet 39: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet --weights weights_folder/39_kimianet
printf "Knet 39: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet --weights weights_folder/39_kimianet --measure remove
printf "Knet 39: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet --weights weights_folder/39_kimianet --measure all


printf "Resnet 40: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/40_resnet --rewrite
printf "Resnet 40: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/40_resnet
printf "Resnet 40: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/40_resnet --measure remove
printf "Resnet 40: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/40_resnet --measure all 
