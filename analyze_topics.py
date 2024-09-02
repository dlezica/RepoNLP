# %%
# IMPORTS
from network.repository import Repository
from utils.eda_utils import load_info, eda, prepare_info_for_model
from pysentimiento import create_analyzer
from data.DocumentDAO import News, data_index
from data.TopicDAO import TopicKeyword, Topic
from utils.utils import SPANISH_STOPWORDS

from dateutil.parser import parse
from datetime import datetime
import spacy

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

#//////////////////////////////////////////////////////////////////////////////////////////
# %% 
# SE DEFINEN LAS FUNCIONES QUE SERÁN DE UTILIDAD
#/-------------------
def load_dataset(dataset):
    df = load_info(dataset = dataset)
    eda(df = df)
    data, kw, entities = prepare_info_for_model(
        df = df,
        first_n_elements = number_of_news_to_analyze
    )

    print("\nExploration After processing the dataframe")
    print("documento 1")
    print(data[0])
    print("primeras 10 kws")
    print(kw[:10])
    print("primeras 10 entidades")
    print(entities[:10])

    return df, data, kw, entities

#/-------------------
def geneate_topics_for_the_day(data, all_tokens):
    tokenizer = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=SPANISH_STOPWORDS,
        lowercase=True,
        vocabulary=all_tokens,
    )
    tokenizer.fit(data)

    model = BERTopic(
        language='spanish',
        calculate_probabilities=False,
        embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
        umap_model=UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine'),
        hdbscan_model=HDBSCAN(min_cluster_size=7, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
        vectorizer_model=tokenizer,
        ctfidf_model=ClassTfidfTransformer(),
        verbose=True,
        min_topic_size=10
    )

    topics, probs = model.fit_transform(data)

    print(len(topics))
    print(len(probs))
    print(len(data))

    print(len(model.get_topics()))
    print(model.get_topics())
    print(topics)

    print(probs)

    return topics, probs, model

#/-------------------
def save_day_batch_documents_to_db(repository, df, model, sentiment_analyzer, date):
    print("starting to save docs")
    for idx in range(0, number_of_news_to_analyze):
        embedding = list(model.embedding_model.embed(df['text'][idx]))
        sentiment_analysis = sentiment_analyzer.predict(
            df['text'][idx]
        ).output

        news = News(
            id = embedding,
            created_at = date,
            text = str(df['text'][idx]),
            entities = str(df['entities'][idx]),
            sentiment_analysis = sentiment_analysis
        )
        news.save(using=repository.get_client())
        print(idx)

    print("end saving docs")

#/-------------------

def get_topic_name(keywords):
    return ', '.join([k for k, s in keywords[:4]])

def save_model_to_db(repository, model):
    print("starting to save topics")
    for topic in model.get_topics().keys():
        if topic > -1:
            s = Topic.search().query("match", index=topic)
            is_existing_topic = s.execute().to_dict()['hits']['total']['value']

            if is_existing_topic == 0:
                keywords = model.topic_representations_[topic]
                topic_keywords = [TopicKeyword(name=k, score=s) for k, s in keywords]

                topic_doc = Topic(
                    vector = list(model.topic_embeddings_[topic + 1]),
                    similarity_threshold = 0.7,
                    created_at = datetime.now(),
                    index = topic,
                    keywords = topic_keywords,
                    name = get_topic_name(keywords),
                )

                topic_doc.save(using=repository.get_client())
                print(topic)

    print("end saving topics")

#/-------------------
def analyze_single_piece_of_news(model, dataset=1, document_number=244):
    df = load_info(dataset = ds_list[dataset])
    single_piece_of_news = df['text'][document_number]
    topic, probs = model.transform(single_piece_of_news)
    document = nlp(single_piece_of_news)
    return single_piece_of_news, topic, probs, document.ents

#/-------------------
def search_documents(client, date_from="2024-07-09", date_to="2024-07-10", qty = 3000):
    '''
    query = {
        "query": {
            "range": {
                "created_at": {
                    "gte": date_from,
                    "lte": date_to
                }
            }
        }
    }

    response = client.search(index=data_index, body=query, size=3000)  # size es opcional, ajusta según lo necesario
    documents = response['hits']['hits']
    
    '''
    
    query = {
        "size":qty,
        "query": {
            "range": {
                "created_at": {
                    "gte": date_from,
                    "lte": date_to
                }
            }
        }
    }

    s = News.search().from_dict(query)
    documents = s.execute().to_dict()['hits']['hits']
    
    docs = []
    timestamps = []
    for doc in documents:
        docs.append(doc['_source']['text'])
        timestamps.append(doc['_source']['created_at'][0:10])

    print(len(docs))
    print(len(timestamps))
    print(docs[3])
    print(timestamps[0])

    return docs, timestamps

#//////////////////////////////////////////////////////////////////////////////////////////
# %%
# SE DEFINEN DATASETS, REPOSITORIO, SENTIMENT ANALYZER, DAOs y NER

# Se trabaja con 1500 noticias de cada dataset porque acelera el entrenamiento y al no tener
# GPU, mi maquina tira timeout si trabajo con el dataset entero.

# Probe usar OnlineTopicModeling y partial_fit, pero me fuerza a fijar al cantidad de topics
# por cada iteracion y eso me pareció sub optimo eso para este ejercicio
number_of_news_to_analyze=1500

ds_list = [
    "jganzabalseenka/news_2024-07-09_24hs",
    "jganzabalseenka/news_2024-07-10_24hs",
    "jganzabalseenka/news_2024-07-11_24hs",
    "jganzabalseenka/news_2024-07-12_24hs",
    "jganzabalseenka/news_2024-07-13_24hs"
]

sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
repository = Repository()
Topic.init(using=repository.get_client())
News.init(using=repository.get_client())
nlp = spacy.load('es_core_news_md')


#//////////////////////////////////////////////////////////////////////////////////////////
# %%
# SETUP INICIAL DIA ZERO, PRIMER BATCH DE NOTICIAS Y GUARDADO DE DOCS Y TOPICS
day_zero = "2024-07-09" 

day_zero_df, data, kw, entities = load_dataset(dataset = ds_list[0])
topics, probs, model = geneate_topics_for_the_day(
    data=data, 
    all_tokens=list(set(kw + entities))
)
save_day_batch_documents_to_db(repository, day_zero_df, model, sentiment_analyzer, day_zero)
save_model_to_db(repository, model)


#//////////////////////////////////////////////////////////////////////////////////////////
# %%
# CASO 1: LLEGA UNA SOLA NOTICIA Y SE ANALIZA
# PARA ESTE CASO UTILIZAREMOS CUALQUIER DOCUMENTO DE CUALQUIER DATASET

dataset_number = 1 # change the dataset you want to check (form:0 - to 4)
document_number = 244 # change the doc inside the dataset selected (from: 0 - to: datasetLength)

single_piece_of_news, topic, probs, ents = analyze_single_piece_of_news(
    dataset=dataset_number, 
    document_number=document_number,
    model = model
)

print(single_piece_of_news)
print("Topic nº: " + str(topic[0]))
print("Prob: " + str(probs))
print("Topic representation - keywords: " + str(model.topic_representations_[topic[0]]))
print("Sentiment analysis: " + sentiment_analyzer.predict(single_piece_of_news).output)
print("Entities: " + str(ents))


#//////////////////////////////////////////////////////////////////////////////////////////
# %%
# CASO 2: SE ANALIZA UN DIA ENTERO NUEVO DE NOTICIAS (DIA 0 y DIA 1)
day_one = "2024-07-10"

day_one_df, data_d1, kw_d1, entities_d1= load_dataset(dataset = ds_list[1])
topics_d1, probs_d1, model_d1 = geneate_topics_for_the_day(
    data=data_d1, 
    all_tokens=list(set(kw_d1 + entities_d1))
)

print("topic quantity pre merge: " + str(len(model.get_topics())))
merged_model = BERTopic.merge_models([model, model_d1], min_similarity=0.85)
save_day_batch_documents_to_db(repository, day_one_df, merged_model, sentiment_analyzer, day_one)
save_model_to_db(repository, merged_model)

# %%
# EDA over merged models for day 0 and 1
print(len(model.get_topics()))
print(len(model_d1.get_topics()))
print(len(merged_model.get_topics()))

topic_info_1 = model.get_topic_info()
topic_info_2 = model_d1.get_topic_info()
merged_topic_info = merged_model.get_topic_info()

print(topic_info_1.tail(5))
print(topic_info_2.tail(5))
print(merged_topic_info.tail(10))

# %%
elegir_nombre_de_topic_de_model_d1 = "46_cámara_diputados_senado_oposición"
selected_abstracts = [item[0] for item in zip(data_d1, topics_d1) if item[1] == topic_info_2.loc[topic_info_2["Name"] == elegir_nombre_de_topic_de_model_d1, "Topic"].values[0]]
print(selected_abstracts)
print(model_d1.transform(selected_abstracts))
print(merged_model.transform(selected_abstracts))



#//////////////////////////////////////////////////////////////////////////////////////////
# %%
# CASO 3: SE ANALIZA UN DIA ENTERO NUEVO DE NOTICIAS (DIA 0, DIA 1 y DIA 2)
day_two = "2024-07-11"

day_two_df, data_d2, kw_d2, entities_d2= load_dataset(dataset = ds_list[2])
topics_d2, probs_d2, model_d2 = geneate_topics_for_the_day(
    data=data_d2, 
    all_tokens=list(set(kw_d2 + entities_d2))
)
print("topic quantity pre merge: " + str(len(merged_model.get_topics())))
merged_model = BERTopic.merge_models([merged_model, model_d2], min_similarity=0.85)
save_day_batch_documents_to_db(repository, day_two_df, merged_model, sentiment_analyzer, day_two)
save_model_to_db(repository, merged_model)

# %%
# EDA over merged models for day 0 and 1
print(len(model.get_topics()))
print(len(model_d1.get_topics()))
print(len(model_d2.get_topics()))
print(len(merged_model.get_topics()))

topic_info_2 = model_d2.get_topic_info()
merged_topic_info = merged_model.get_topic_info()

print(topic_info_2.tail(5))
print(merged_topic_info.tail(10))

# %%
elegir_nombre_de_topic_de_model_d2 = "49_toneladas_exportaciones_producción_maíz"

selected_abstracts = [item[0] for item in zip(data_d2, topics_d2) if item[1] == topic_info_2.loc[topic_info_2["Name"] == elegir_nombre_de_topic_de_model_d2, "Topic"].values[0]]
print(selected_abstracts)
print(model_d2.transform(selected_abstracts))
print(merged_model.transform(selected_abstracts))




#//////////////////////////////////////////////////////////////////////////////////////////
# %%
# CASO 4: SE ANALIZA UN DIA ENTERO NUEVO DE NOTICIAS (DIA 0, DIA 1, DIA 2 y DIA 3)
day_three = "2024-07-12"

day_three_df, data_d3, kw_d3, entities_d3= load_dataset(dataset = ds_list[3])
topics_d3, probs_d3, model_d3 = geneate_topics_for_the_day(
    data=data_d3, 
    all_tokens=list(set(kw_d3 + entities_d3))
)
print("topic quantity pre merge: " + str(len(merged_model.get_topics())))
merged_model = BERTopic.merge_models([merged_model, model_d3], min_similarity=0.85)
save_day_batch_documents_to_db(repository, day_three_df, merged_model, sentiment_analyzer, day_three)
save_model_to_db(repository, merged_model)

# %%
# EDA over merged models for day 0 and 1
print(len(model.get_topics()))
print(len(model_d1.get_topics()))
print(len(model_d2.get_topics()))
print(len(model_d3.get_topics()))
print(len(merged_model.get_topics()))

topic_info_2 = model_d3.get_topic_info()
merged_topic_info = merged_model.get_topic_info()

print(topic_info_2.tail(5))
print(merged_topic_info.tail(10))

# %%
elegir_nombre_de_topic_de_model_d3 = ""

selected_abstracts = [item[0] for item in zip(data_d3, topics_d3) if item[1] == topic_info_2.loc[topic_info_2["Name"] == elegir_nombre_de_topic_de_model_d3, "Topic"].values[0]]
print(selected_abstracts)
print(model_d3.transform(selected_abstracts))
print(merged_model.transform(selected_abstracts))



#//////////////////////////////////////////////////////////////////////////////////////////
# %%
# CHECK TOPICS OVER TIME
'''
Se va por esta solucion de re-entrenar el class tfidf y tomar el vectorizer porque el modelo que se genera despues del merge no hace un merge de estas cosas. solo de topicos.
Para este caso se está siguiendo la solucion propuesta por el creador de la lib en este issue (que aun sigue abierto)

https://github.com/MaartenGr/BERTopic/issues/1700

'''

import pandas as pd

# Select date from and two here
from_date = day_zero
to_date = day_two

docs, timestamps = search_documents(repository.get_client(), from_date, to_date, qty= 4500)

documents = pd.DataFrame(
    {
        "Document": docs,
        "ID": range(len(docs)),
        "Topic": merged_model.topics_,
        "Image": None
    }
)
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})

merged_model.vectorizer_model = model.vectorizer_model

c_tf_idf, _ = merged_model._c_tf_idf(documents_per_topic)
merged_model.c_tf_idf_ = c_tf_idf

merged_model.transform(docs)

topics_over_time = merged_model.topics_over_time(
    docs=docs, 
    timestamps=timestamps
)
merged_model.visualize_topics_over_time(topics_over_time)



#//////////////////////////////////////////////////////////////////////////////////////////

# %%
