# DeepNup

1.Encoding
===
DataNup_1, 2 and 3 are all stored in data.zip, you can unzip the file using the following command:
```
unzip data.zip -d data
```
* **-d data:** The name of the folder you unzipped

Encode the data with the following command：
```
$python Data_encoded.py -p Fasta_File_Path -f Fasta_filename -o Output_file_Path
```
* **-p Fasta_File_Path:** the path of fasta files. If your unzip folder name is data, the path is data/DataNup_1(or DataNup_2, DataNup_3);
* **-f Fasta_filename:** all fasta file names can be found in the data folder;
* **-o Output_file_Path:** you can use this parameter to pass the path where the encoded data will be saved.


2.Train the model
===
Train DeepNup with the following command:
```
$python training.py -p Pickle_file_Path -o Output_file_Path -e Experiments_Name
```
* **-p Pickle_file_Path:** the path of the pickle file, it is same as the Output_file_Path in the `1.`;
* **-o Output_file_Path:** you can use this parameter to pass the path where the model will be saved;
* **-e Experiments_Name:** you can use this parameter to pass the name of this experiment.


3.Predict and get results
===
Predict DeepNup with the following command:
```
$python predict.py -p Model_file_Path -e Experiments_Name
```
* **-p Model_file_Path:** the path where the model was previously saved, it is same as Output_file_Path in `2.`;
* **-e Experiments_Name:** the name of this experiment, it is same as Experiments_Name in `2.`.
