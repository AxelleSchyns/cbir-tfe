
#!/usr/bin/env bash

printf "12 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/87_resnet --rewrite 

printf "12 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/87_resnet --measure stat

printf "12 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/87_resnet --measure all

printf "12 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/87_resnet --measure weighted



printf "16 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/88_resnet --rewrite

printf "16 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/88_resnet --measure stat

printf "16 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/88_resnet --measure all

printf "16 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/88_resnet --measure weighted


printf "11 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/11_new_resnet --rewrite --dr_model

printf "11 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/11_new_resnet --measure stat --dr_model

printf "11 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/11_new_resnet --measure all --dr_model

printf "11 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/11_new_resnet --measure weighted --dr_model