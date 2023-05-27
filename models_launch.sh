
#!/usr/bin/env bash



printf "44 vgg16 \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vgg16 --weights weights_folder/44_vgg16.pth --rewrite --num_features 25088

printf "44 vgg16: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/44_vgg16.pth --measure stat --num_features 25088




printf "95 vgg16 \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vgg16 --weights weights_folder/95_vgg16.pth --rewrite --num_features 25088

printf "95 vgg16: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/95_vgg16.pth --measure stat --num_features 25088




printf "96 vgg16 \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vgg16 --weights weights_folder/96_vgg16.pth --rewrite --num_features 25088

printf "96 vgg16: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/96_vgg16.pth --measure stat --num_features 25088

printf "96 vgg16: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/96_vgg16.pth --measure all --num_features 25088





printf "66 auto \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor auto --weights weights_folder/66_auto --rewrite --num_features 16

printf "66 auto: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor auto --weights weights_folder/66_auto --measure stat --num_features 16

printf "66 auto: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor auto --weights weights_folder/66_auto --measure all --num_features 16

printf "66 auto: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor auto --weights weights_folder/66_auto --measure weighted --num_features 16