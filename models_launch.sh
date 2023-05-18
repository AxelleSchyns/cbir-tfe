
#!/usr/bin/env bash
printf "24 Densenet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor densenet --weights weights_folder/24_densenet --rewrite

printf "24 densenet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/24_densenet --measure stat

printf "24 densenet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/24_densenet --measure all

printf "24 densenet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor densenet --weights weights_folder/24_densenet --measure weighted


printf "15 effnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor effnet --weights weights_folder/15_effnet --rewrite

printf "15 effnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet --weights weights_folder/15_effnet --measure stat

printf "15 effnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet --weights weights_folder/15_effnet --measure all

printf "15 effnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor effnet --weights weights_folder/15_effnet --measure weighted


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


printf "85 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/85_resnet --rewrite

printf "85 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/85_resnet --measure stat

printf "85 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/85_resnet --measure all

printf "85 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/85_resnet --measure weighted


printf "18 vgg16 \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor vgg16 --weights weights_folder/18_vgg16 --rewrite

printf "18 vgg16: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/18_vgg16 --measure stat

printf "18 vgg16: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/18_vgg16 --measure all

printf "18 vgg16: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor vgg16 --weights weights_folder/18_vgg16 --measure weighted