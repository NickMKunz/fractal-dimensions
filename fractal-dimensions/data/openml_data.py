import numpy as np
import openml



# get numerical data from openml data set
def openml_data(id):
    dataset = openml.datasets.get_dataset(id, download_data=True, download_qualities=False, download_features_meta_data=False)
    df, *_ = dataset.get_data()
    df = df.select_dtypes(include=np.number)
    return df


###--- LIST OF INTERESTING DATA SETS FROM OPENML (WITH INTRINSIC DIMENSION COMPUTED IN https://arxiv.org/abs/2109.02596) ---###

# SensorDataResource: ID=23383 (int_dim=1 in emb_dim=25)
# hill-valley: ID=1479 (int_dim=1 in emb_dim=100)
# pokerhand: ID=155 (int_dim=3 in emb_dim=5)
# satellite_image: ID=294 (int_dim=2 in emb_dim=36)
# waveform-5000: ID=60 (int_dim=7 in emb_dim=40)
# JapaneseVowels: ID=375 (int_dim=6 in emb_dim=14)
# electrical-grid-stability: ID=43007 (int_dim=10 in emb_dim=13)