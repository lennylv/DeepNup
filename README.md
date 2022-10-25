# DeepNup

1.Encoding
===
DataNup_1, 2 and 3 are all stored in data.zip, you can unzip the file using the following command:
```
unzip data.zip -d data
```

Encode the data with the following command：
```
$python Data_encoded.py -p Fasta_File_Path -f Fasta_filename -o Output_file_Path -n Output Nucleosome_filename -l Output Linker_filename
```

2.Train the model
===
Train DeepNup with the following command:
```
$python training.py -p Pickle_file_Path -o Output_file_Path -e Experiments Name
```

3.Predict and get results
===
Predict DeepNup with the following command:
```
$python predict.py -p Model file_Path -e Experiments_Name
```
