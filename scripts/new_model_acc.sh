#!/usr/bin/env bash

conda activate tfe

python database/test_accuracy.py --path /home/lab/Documents/Axelle/cytomine/Data/validation --extractor CHANGE --weights weights_folder/CHANGE  --measure all

python database/test_accuracy.py --path /home/lab/Documents/Axelle/cytomine/Data/validation --extractor CHANGE --weights weights_folder/CHANGE

python database/test_accuracy.py --path /home/lab/Documents/Axelle/cytomine/Data/validation --extractor CHANGE --weights weights_folder/CHANGE --measure remove

python database/test_accuracy.py --path /home/lab/Documents/Axelle/cytomine/Data/validation --extractor CHANGE --weights weights_folder/CHANGE --name CHANGE --excel_path results_class.xslx --measure all

