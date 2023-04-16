# Compétition: LeafNothingBehind

membres de l'équipe:
- Diouf Assane
- Han Cléa
- Labeyrie Yanis
- Zabban Adrien

# Contenu

## Submission

Dossier à rendre à Jules. Contient:
- dossier `checkpoint` avec le ou les poids du model
- dossier `src` qui contient les programme python tel que le dataloader, le model, le train, le test, etc...
- le programme `main.py` qui va être éxécuter pas Jules, dans lequel on peut lui passer les argument suivant:
  - `mode`: this argument will be given the value 'infer' or 'train'
  - `csv_path`: we will use this argument to give your code the path to the CSV file for the test data. Those CSV files will obey the same format as the train dataset, and the folder structure of the dataset will also be the same
  - `save_infers_under`: we will use this argument to give your code the path to the folder where to save the results.
- le fichier `requirements.txt` contenant l'ensemble des packages utilisés ainsi que leur version
- le fichier package GDAL

## test_submission

dossier dans lequel on teste pour voir si la submission marche. Elle contient des donnes de tests.
Pour tester la submission avec python, faites:
```bash
python main.py --mode infer --csv_path ..\test_submission\test_data.csv --save_infers_under ..\test_submission\result
```

Pour tester la submission avec le script bash, faites:
```bash
bash server_script.sh ../test_submission/test_data.csv ../test_submission/result submission_16_april.zip
```


## Dans le utils
- `names_on_drive.txt`: listes des noms des fichiers s2 (au temps $t$) qui sont sur le drive. *last update: 16 avril*
- `deplace_data.py`: programme python qui copie des fichiers et les mettents dans un autre dossier
- `check_images.py`: fait la liste des noms des autres images que l'on peut utiliser
