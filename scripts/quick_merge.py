import pandas as pd
import os
dataset_dir = "./data/AudioSet"
df = pd.read_csv(os.path.join(dataset_dir,"audioset_train_strong.tsv"),sep = "\t")
df2 = pd.read_csv(os.path.join(dataset_dir,"mid_to_display_name.tsv"),sep = "\t",names = ["label","string_label"])
df = df.join(df2.set_index('label'), on='label')
df.to_csv(os.path.join(dataset_dir,"audioset_train_strong_with_label.tsv"),sep = "\t", index= False)