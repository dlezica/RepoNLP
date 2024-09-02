from datasets import load_dataset
import pandas as pd
from utils.utils import SPANISH_STOPWORDS
from tqdm import tqdm
import unicodedata
from functools import wraps
import re


class Cleaning_text:
    '''
    Limpiar elementos no deseados del texto 
    '''

    def __init__(self):
        # Definir los caracteres Unicode no deseados
        self.unicode_pattern    = ['\u200e', '\u200f', '\u202a', '\u202b', '\u202c', '\u202d', '\u202e', '\u202f']
        self.urls_pattern       = re.compile(r'http\S+')
        self.simbols_chars      = r"""#&’'"`´“”″()[]*+,-.;:/<=>¿?!¡@\^_{|}~©√≠"""                 # Lista de símbolos a eliminar
        self.simbols_pattern    = re.compile(f"[{re.escape(self.simbols_chars)}]")    
        self.escape_pattern     = ['\n', '\t', '\r']
        
    def _clean_decorator(clean_func):
        @wraps(clean_func)
        def wrapper(self, input_data):
            def clean_string(text):
                return clean_func(self, text)

            if isinstance(input_data, str):
                return clean_string(input_data)
            elif isinstance(input_data, list):
                return [clean_string(item) for item in input_data]
            else:
                raise TypeError("El argumento debe ser una cadena o una lista de cadenas.")
        return wrapper

    @_clean_decorator
    def unicode(self, text):
        for pattern in self.unicode_pattern:
            text = text.replace(pattern, ' ')
        return text

    @_clean_decorator
    def urls(self, text):
        return self.urls_pattern.sub(' ', text)
    
    @_clean_decorator
    def simbols(self, text):
        return self.simbols_pattern.sub(' ', text)

    @_clean_decorator
    def accents_emojis(self, text):
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
    @_clean_decorator
    def escape_sequence(self, text):
        for pattern in self.escape_pattern:
            text = text.replace(pattern, ' ').strip()
        return text
    
    @_clean_decorator
    def str_lower(self, text):
        return text.lower()
    

def clean_all(entities, accents=True, lower=True) -> list:
    cleaner = Cleaning_text()

    entities_clean = []
    for ent in entities:
        clean_txt = cleaner.unicode(ent)
        clean_txt = cleaner.urls(clean_txt)
        clean_txt = cleaner.simbols(clean_txt)
        
        if accents:
            clean_txt = cleaner.accents_emojis(clean_txt)

        clean_txt = cleaner.escape_sequence(clean_txt)

        if lower:
            clean_txt = cleaner.str_lower(clean_txt)
        
        entities_clean.append(" ".join(clean_txt.split()))
            
    return entities_clean

def __sanitize_info(df):
    resut_list = [df[col].hasnans for col in df]
    for result in resut_list:
        if result != False:
            return False
    return True 

def __get_kws(df, first_n_elements=10000):
    key_1 = df['keywords'][:first_n_elements]
    key_1_set = list(set([ keyw.lower() for sublista in key_1 for keyw in sublista ]))
    key_1_clean = clean_all(key_1_set, accents=False)
    key_1_unique = [ word for word in key_1_clean if word not in SPANISH_STOPWORDS]
    return key_1_unique

def __get_entities(df, first_n_elements=10000):
    ent_1 = df['entities'][:first_n_elements]
    ent_1_set = list(set([ ent.lower() for sublista in ent_1 for ent in sublista ]))
    ent_1_clean = clean_all(ent_1_set, accents=False)
    ent_1_unique = [ word for word in ent_1_clean if word not in SPANISH_STOPWORDS]
    return ent_1_unique

def load_info(dataset):
    ds = load_dataset(dataset, split="train")
    df = pd.DataFrame(ds)
    if __sanitize_info(df):
        return df    
    return "The dataset contain nulls"

def eda(df):
    print("EDA exploration befor processing the dataframe")
    print("Kw: " +str(df.iloc[0]['keywords']))
    print("Entities: " + str(df.iloc[0]['entities_transformers']))
    print("canidad de documentos: " + str(len(df)))
    print("canidad de documentos en lista: " + str(len(df)))
    print(df.head())

def prepare_info_for_model(df, first_n_elements=10000):
    clean_data = Cleaning_text()
    data = list(df['text'][:first_n_elements])

    proc_data_1 = []
    for data_in in tqdm(data):
        aux = clean_data.unicode(data_in)
        aux = clean_data.urls(aux)
        aux = clean_data.simbols(aux)
        aux = clean_data.escape_sequence(aux)
        aux = " ".join([ word for word in aux.split() if word.lower() not in SPANISH_STOPWORDS])
        proc_data_1.append(aux)

    kw = __get_kws(df, first_n_elements)
    entities = __get_entities(df, first_n_elements)
    return proc_data_1 , kw, entities