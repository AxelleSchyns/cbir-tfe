
#!/usr/bin/env bash

printf "98 deit \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor deit --weights weights_folder/98_deit --rewrite

printf "98 deit: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor deit --weights weights_folder/98_deit --measure stat

printf "98 deit: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor deit --weights weights_folder/98_deit --measure all

printf "98 deit: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor deit --weights weights_folder/98_deit --measure weighted




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




printf "86 resnet\n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/86_resnet --rewrite

printf "86 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/86_resnet --measure stat

printf "86 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/86_resnet --measure all

printf "86 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/86_resnet --measure weighted




printf "87 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/87_resnet --rewrite

printf "87 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/87_resnet --measure stat

printf "87 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/87_resnet --measure all

printf "87 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/87_resnet --measure weighted





printf "88 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/88_resnet --rewrite

printf "88 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/88_resnet --measure stat

printf "88 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/88_resnet --measure all

printf "88 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/88_resnet --measure weighted




printf "89 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/89_resnet --rewrite

printf "89 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/89_resnet --measure stat

printf "89 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/89_resnet --measure all

printf "89 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/89_resnet --measure weighted




printf "90 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/90_resnet --rewrite

printf "90 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/90_resnet --measure stat

printf "90 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/90_resnet --measure all

printf "90 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/90_resnet --measure weighted




printf "91 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/91_resnet --rewrite

printf "91 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/91_resnet --measure stat

printf "91 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/91_resnet --measure all

printf "91 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/91_resnet --measure weighted




printf "92 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/92_resnet --rewrite

printf "92 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/92_resnet --measure stat

printf "92 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/92_resnet --measure all

printf "92 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/92_resnet --measure weighted




printf "93 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/93_resnet --rewrite

printf "93 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/93_resnet --measure stat

printf "93 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/93_resnet --measure all

printf "93 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/93_resnet --measure weighted




printf "94 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights weights_folder/94_resnet --rewrite

printf "94 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/94_resnet --measure stat

printf "94 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/94_resnet --measure all

printf "94 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights weights_folder/94_resnet --measure weighted




printf "61 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights 61_resnet --rewrite

printf "61 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights 61_resnet --measure stat

printf "61 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights 61_resnet --measure all

printf "61 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights 61_resnet --measure weighted




printf "63 resnet \n"
python database/add_images.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/test --extractor resnet --weights 63_resnet --rewrite

printf "63 resnet: random \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights 63_resnet --measure stat

printf "63 resnet: all \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights 63_resnet --measure all

printf "63 resnet: weighted \n"
python database/test_accuracy.py --path /home/labarvr4090/Documents/Axelle/cytomine/Data/validation --extractor resnet --weights 63_resnet --measure weighted

