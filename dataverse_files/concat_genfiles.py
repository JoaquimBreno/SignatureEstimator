import pandas as pd
dataset1 = pd.read_csv('data_train_4_classes.csv')

dataset2 = pd.read_csv('art.csv')
dataset_concatenado = pd.concat([dataset1, dataset2], ignore_index=True)
dataset_concatenado.to_csv('data_train_art_4_classes.csv', index=False)