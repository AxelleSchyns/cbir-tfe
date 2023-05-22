
#!/usr/bin/env bash

printf "84 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/84_resnet --rewrite

printf "84 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/84_resnet --measure stat

printf "84 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/84_resnet --measure all

printf "84 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/84_resnet --measure weighted



printf "14 kimianet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor knet --weights weights_folder/14_kimianet --rewrite

printf "14 kimianet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet --weights weights_folder/14_kimianet --measure stat

printf "14 kimianet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet --weights weights_folder/14_kimianet --measure all

printf "14 kimianet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor knet --weights weights_folder/14_kimianet --measure weighted



printf "29 vision \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vision --weights weights_folder/29_vision --rewrite

printf "29 vision: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vision --weights weights_folder/29_vision --measure stat

printf "29 vision: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vision --weights weights_folder/29_vision --measure all

printf "29 vision: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vision --weights weights_folder/29_vision --measure weighted



printf "32 cvt \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor cvt --weights weights_folder/32_cvt --rewrite

printf "32 cvt: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor cvt --weights weights_folder/32_cvt --measure stat

printf "32 cvt: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor cvt --weights weights_folder/32_cvt --measure all

printf "32 cvt: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor cvt --weights weights_folder/32_cvt --measure weighted



printf "97 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/97_resnet --rewrite --dr_model

printf "97 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/97_resnet --measure stat --dr_model

printf "97 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/97_resnet --measure all --dr_model

printf "97 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/97_resnet --measure weighted --dr_model



printf "61 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/61_resnet --rewrite

printf "61 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/61_resnet --measure stat

printf "61 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/61_resnet --measure all

printf "61 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/61_resnet --measure weighted



printf "63 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/63_resnet --rewrite

printf "63 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/63_resnet --measure stat

printf "63 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/63_resnet --measure all

printf "63 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/63_resnet --measure weighted



printf "18 vgg16 \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vgg16 --weights weights_folder/18_vgg16.pth --rewrite

printf "18 vgg16: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/18_vgg16.pth --measure stat

printf "18 vgg16: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/18_vgg16.pth --measure all

printf "18 vgg16: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/18_vgg16.pth --measure weighted



printf "64 vae \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor VAE --weights weights_folder/64_VAE --rewrite --num_features 3840

printf "64 vae: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor VAE --weights weights_folder/64_VAE --measure stat --num_features 3840



printf "82 auto \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor auto --weights weights_folder/82_auto --rewrite --num_features 3072

printf "82 auto: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor auto --weights weights_folder/82_auto --measure stat --num_features 3072




