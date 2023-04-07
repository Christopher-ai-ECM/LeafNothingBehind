# Compétition: LeafNothingBehind

membres de l'équipe:
- Diouf Assane
- Han Cléa
- Labeyrie Yanis
- Zabban Adrien

# Contenu

- `names_on_drive.txt`: listes des noms des fichiers s2 (au temps $t$) qui sont sur le drive. *last update: 7 avril*
- `have_to_add.txt`: listes de noms de fichiers s2 (au temps $t$) tel que:
  - leur mask assosié ne contiennet pas de $0$ (qui signifie `NO_DATA`)
  - leur nom de fichier sont aussi dans s1, s2-mask, ainsi que les noms des images précétendes (au temps $t-1$ et $t-2$)
  - il ne sont pas déjà dans `names_on_drive.txt`, donc qui ne sont pas déja dans le drive
- `deplace_data.py`: programme python qui copie des fichiers et les mettents dans un autre dossier
- `check_images.py`: fait la liste des noms des autres images que l'on peut utiliser
