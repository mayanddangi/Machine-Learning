Code for all the task is contains in jupyter notebook: Assignment 1.ipynb

main.py file is to train the model and gives output in .csv file

python main.py --train_path=<path to train file> --val_path=<path to validation file> --test_path=<path to test file> --out_path=<path to generated output scores> --section=<1 or 2 or 5>

argument should be provided in exact format as shown above, = sign is necessary as this is used to seprate the source path
For example: main.py --train_path=train.csv --val_path=validation.csv --test_path=train.csv --out_path=./output --section=1
DONT DO: main.py --train_path="train.csv" --val_path="validation.csv" --test_path="train.csv" --out_path="./output" --section="1"

If you want to save the output file in current directory just write <.>


linear.py : Contains model to train for linear regression
classification.fy : Contains model for multiclass classification
ridge.py: Contains model for ridge classification

All the above mentioned file should contained at the location of main.py

