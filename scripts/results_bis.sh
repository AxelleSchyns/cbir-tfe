printf "Densenet 21: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor densenet --weights weights_folder/21_densenet --rewrite --dr_model
printf "Densenet 21: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/21_densenet --dr_model
printf "Densenet 21: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/21_densenet --measure remove --dr_model
printf "Densenet 21: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/21_densenet --measure all --dr_model

printf "Densenet 37: random \n"
python database/classification_acc.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/37_densenet
printf "Densenet 37: remove \n"
python database/classification_acc.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/37_densenet --measure remove
printf "Densenet 37: all \n"
python database/classification_acc.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/37_densenet --measure all 

printf "Resnet 40: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/40_resnet --rewrite
printf "Resnet 40: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/40_resnet
printf "Resnet 40: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/40_resnet --measure remove
printf "Resnet 40: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/40_resnet --measure all 


printf "Autoencoder 34: indexing \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet18 --weights weights_folder/34_autoencoder.pth --rewrite --num_features 25088
printf "Autoencoder 34: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet18 --weights weights_folder/34_autoencoder.pth --num_features 25088
printf "Autoencoder 34: remove \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet18 --weights weights_folder/34_autoencoder.pth --measure remove --num_features 25088
printf "Autoencoder 34: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet18 --weights weights_folder/34_autoencoder.pth --measure all --num_features 25088
