
#!/usr/bin/env bash

#printf "50 resnet \n"
#python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/50_resnet --rewrite --generalise 3

#printf "50 resnet: all \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/50_resnet --measure all --generalise 3

#printf "50 resnet: weighted \n"
#python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/50_resnet --measure weighted --generalise 3


#printf "82 auto \n"
#python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor auto --weights weights_folder/71_auto --rewrite 


python database/rec_images.py --path  /home/labarvr4090/Documents/Axelle/cytomine/Data/test/cells_no_aug_0/728755_748504.png --extractor resnet18 --weights weights_folder/102_resnet18  --namefig "rec_102_resnet18" 

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