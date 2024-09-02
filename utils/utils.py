from matplotlib import pyplot as plt
import pandas as pd

SPANISH_STOPWORDS = list(pd.read_csv('utils/spanish_stop_words.csv' )['stopwords'].values)
