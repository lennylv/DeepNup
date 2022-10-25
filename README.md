# DeepNup

1.Encoding
---
DataNup_1, 2 and 3 are all stored in data.zip, you can unzip the file using the following command:
```
unzip data.zip -d data
```
* **-d data:** the folder name you unzip.

Encode the data with the following command：
```
$python Data_encoded.py -p Fasta_file_Path -f Fasta_filename -o Output_file_Path
```
* **-p Fasta_file_Path:**  the path of fasta file of raw DNA sequence;
* **-f Fasta_filename:** the file name of fasta file of raw DNA sequence;
* **-o Output_file_Path:** the path where the encoding data of DNA sequences is saved.


2.Train the model
---
Train DeepNup with the following command:
```
$python training.py -p Pickle_file_Path -o Output_file_Path -e Experiments_Name
```
* **-p Pickle_file_Path:** the path of the pickle file;
* **-o Output_file_Path:** the path where the model is saved;
* **-e Experiments_Name:** the name of the folder where one model for a specific task is saved.


3.Predict and get results
---
Predict DeepNup with the following command:
```
$python predict.py -p Model_file_Path -e Experiments_Name
```
* **-p Model_file_Path:** the path where the model was previously saved;
* **-e Experiments_Name:** the name of the folder where one model for a specific task is saved.
