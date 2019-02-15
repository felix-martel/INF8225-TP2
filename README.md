# INF8225-TP2
*Paris, Neymar, Nasser, Qatar, voiture très rare*

- les constantes (dimensions des images, nombre de canaux, etc.) et les paramètres (learning rate, nombre d'epochs, batch size) sont définis respectivement dans `constants.py` et `params.py`

- `data` contient trois `torch.utils.data.DataLoader` : `train`, `val` et `test`

- `models/` contient les NN, chacun dans son fichier dédié :
     
     - `cnn.py`
     
     - `fully_connected.py`
     
- `main.py` définit deux fonctions `train` et `eval`
