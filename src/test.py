import pandas as pd
import numpy as np

ANT_NLP_FILE_PATH = '../data/atec_nlp_sim.csv'
ANT_NLP_RESULT_PATH = '../data/result.csv'

origin_data = pd.read_csv(ANT_NLP_FILE_PATH, quoting=3, sep="\t", header=None, names=["id", "sent1", "sent2", "label"],
                          encoding="utf8")
print(origin_data.head())

result_data = pd.read_csv(ANT_NLP_RESULT_PATH, quoting=3, header=0,
                          names=["id", "index", "label"],
                          encoding="utf8")
print(result_data.head())

compare = origin_data['label'] == result_data['label']
print(np.sum(compare) / len(compare))
