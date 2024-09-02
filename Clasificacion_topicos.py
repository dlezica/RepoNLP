<<<<<<< HEAD
# %%
# DEPENDENCIAS
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

# %%
# FUNCIONES UTILES

def cargar_datos(dataset):
    dataframe = load_info(dataset=dataset)
    eda(df=dataframe)
    procesados, palabras_clave, entidades = prepare_info_for_model(
        df=dataframe,
        first_n_elements=numero_de_noticias
    )

    print("\nExploración después de procesar el dataframe")
    print("Documento 1:", procesados[0])
    print("Primeras 10 palabras clave:", palabras_clave[:10])
    print("Primeras 10 entidades:", entidades[:10])

    return dataframe, procesados, palabras_clave, entidades


def generar_topicos_del_dia(textos, tokens):
    vectorizador = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=SPANISH_STOPWORDS,
        lowercase=True,
        vocabulary=tokens,
    )
    vectorizador.fit(textos)

    modelo = BERTopic(
        language='spanish',
        calculate_probabilities=False,
        embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
        umap_model=UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine'),
        hdbscan_model=HDBSCAN(min_cluster_size=7, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
        vectorizer_model=vectorizador,
        ctfidf_model=ClassTfidfTransformer(),
        verbose=True,
        min_topic_size=10
    )

    topicos, probabilidades = modelo.fit_transform(textos)

    print(f"Cantidad de tópicos: {len(topicos)}")
    print(f"Cantidad de probabilidades: {len(probabilidades)}")
    print(f"Cantidad de textos: {len(textos)}")

    print(f"Tópicos del modelo: {modelo.get_topics()}")

    return topicos, probabilidades, modelo


def guardar_documentos_a_db(repo, dataframe, modelo, analizador, fecha):
    print("Iniciando guardado de documentos")
    for i in range(numero_de_noticias):
        vector = list(modelo.embedding_model.embed(dataframe['text'][i]))
        analisis_sentimientos = analizador.predict(dataframe['text'][i]).output

        noticia = News(
            id=vector,
            created_at=fecha,
            text=str(dataframe['text'][i]),
            entities=str(dataframe['entities'][i]),
            sentiment_analysis=analisis_sentimientos
        )
        noticia.save(using=repo.get_client())
        print(f"Documento {i} guardado")

    print("Guardado de documentos finalizado")


def obtener_nombre_de_topico(palabras_clave):
    return ', '.join([p for p, s in palabras_clave[:4]])


def guardar_modelo_a_db(repo, modelo):
    print("Iniciando guardado de tópicos")
    for topico in modelo.get_topics().keys():
        if topico > -1:
            consulta = Topic.search().query("match", index=topico)
            existe_topico = consulta.execute().to_dict()['hits']['total']['value']

            if existe_topico == 0:
                palabras_clave = modelo.topic_representations_[topico]
                topico_palabras = [TopicKeyword(name=k, score=s) for k, s in palabras_clave]

                documento_topico = Topic(
                    vector=list(modelo.topic_embeddings_[topico + 1]),
                    similarity_threshold=0.7,
                    created_at=datetime.now(),
                    index=topico,
                    keywords=topico_palabras,
                    name=obtener_nombre_de_topico(palabras_clave),
                )

                documento_topico.save(using=repo.get_client())
                print(f"Tópico {topico} guardado")

    print("Guardado de tópicos finalizado")


def analizar_noticia_individual(modelo, dataset=1, numero_doc=244):
    dataframe = load_info(dataset=ds_list[dataset])
    noticia = dataframe['text'][numero_doc]
    topico, probabilidades = modelo.transform(noticia)
    documento = nlp(noticia)
    return noticia, topico, probabilidades, documento.ents


def buscar_documentos(cliente, desde="2024-07-09", hasta="2024-07-10", cantidad=3000):
    consulta = {
        "size": cantidad,
        "query": {
            "range": {
                "created_at": {
                    "gte": desde,
                    "lte": hasta
                }
            }
        }
    }

    busqueda = News.search().from_dict(consulta)
    documentos = busqueda.execute().to_dict()['hits']['hits']
    
    textos = []
    fechas = []
    for documento in documentos:
        textos.append(documento['_source']['text'])
        fechas.append(documento['_source']['created_at'][0:10])

    print(f"Documentos encontrados: {len(textos)}")
    print(f"Fechas encontradas: {len(fechas)}")

    return textos, fechas


# %%
# VARIABLES GLOBALES

numero_de_noticias = 1500
ds_list = [
    "jganzabalseenka/news_2024-07-09_24hs",
    "jganzabalseenka/news_2024-07-10_24hs",
    "jganzabalseenka/news_2024-07-11_24hs",
    "jganzabalseenka/news_2024-07-12_24hs",
    "jganzabalseenka/news_2024-07-13_24hs"
]

analizador_sentimientos = create_analyzer(task="sentiment", lang="es")
repositorio = Repository()
Topic.init(using=repositorio.get_client())
News.init(using=repositorio.get_client())
nlp = spacy.load('es_core_news_md')


# %%
# PROCESAMIENTO DE DATOS PARA EL DÍA 0

dia_inicial = "2024-07-09" 

df_dia_cero, textos_cero, palabras_cero, entidades_cero = cargar_datos(dataset=ds_list[0])
topicos_cero, probabilidades_cero, modelo_cero = generar_topicos_del_dia(
    textos=textos_cero, 
    tokens=list(set(palabras_cero + entidades_cero))
)
guardar_documentos_a_db(repositorio, df_dia_cero, modelo_cero, analizador_sentimientos, dia_inicial)
guardar_modelo_a_db(repositorio, modelo_cero)


# %%
# CASO 1: ANALIZAR UNA NOTICIA INDIVIDUAL

dataset_seleccionado = 1
documento_seleccionado = 244

noticia, topico, probabilidades, entidades = analizar_noticia_individual(
    modelo=modelo_cero, 
    dataset=dataset_seleccionado, 
    numero_doc=documento_seleccionado
)

print(noticia)
print(f"Tópico nº: {topico[0]}")
print(f"Probabilidad: {probabilidades}")
print(f"Representación del tópico - Palabras clave: {modelo_cero.topic_representations_[topico[0]]}")
print(f"Análisis de sentimientos: {analizador_sentimientos.predict(noticia).output}")
print(f"Entidades: {entidades}")


# %%
# CASO 2: ANALIZAR UN DÍA COMPLETO DE NOTICIAS (DÍA 0 y DÍA 1)

dia_siguiente = "2024-07-10"

df_dia_uno, textos_uno, palabras_uno, entidades_uno = cargar_datos(dataset=ds_list[1])
topicos_uno, probabilidades_uno, modelo_uno = generar_topicos_del_dia(
    textos=textos_uno, 
    tokens=list(set(palabras_uno + entidades_uno))
)

print(f"Cantidad de tópicos antes de la fusión: {len(modelo_cero.get_topics())}")
modelo_combinado = BERTopic.merge_models([modelo_cero, modelo_uno], min_similarity=0.85)
guardar_documentos_a_db(repositorio, df_dia_uno, modelo_combinado, analizador_sentimientos, dia_siguiente)
guardar_modelo_a_db(repositorio, modelo_combinado)

# EDA de modelos fusionados para los días 0 y 1
print(len(modelo_cero.get_topics()))
print(len(modelo_uno.get_topics()))
print(len(modelo_combinado.get_topics()))

info_topicos_1 = modelo_cero.get_topic_info()
info_topicos_2 = modelo_uno.get_topic_info()
info_topicos_combinados = modelo_combinado.get_topic_info()

print(info_topicos_1.tail(5))
print(info_topicos_2.tail(5))
print(info_topicos_combinados.tail(10))

# Aquí reemplazas el código original con la nueva lógica
nombre_topico_modelo_uno = "46_cámara_diputados_senado_oposición"

# Verifica si el nombre del tópico existe antes de intentar acceder al primer elemento
topico_encontrado = info_topicos_2.loc[info_topicos_2["Name"] == nombre_topico_modelo_uno, "Topic"].values

if topico_encontrado.size > 0:
    resumenes_seleccionados = [item[0] for item in zip(textos_uno, topicos_uno) if item[1] == topico_encontrado[0]]
    print(resumenes_seleccionados)
    print(modelo_uno.transform(resumenes_seleccionados))
    print(modelo_combinado.transform(resumenes_seleccionados))
else:
    print(f"No se encontró ningún tópico con el nombre: {nombre_topico_modelo_uno}")


# %%

# CASO 3: ANALIZAR UN DÍA COMPLETO DE NOTICIAS (DÍA 0, DÍA 1 y DÍA 2)

dia_tres = "2024-07-11"

df_dia_dos, textos_dos, palabras_dos, entidades_dos = cargar_datos(dataset=ds_list[2])
topicos_dos, probabilidades_dos, modelo_dos = generar_topicos_del_dia(
    textos=textos_dos, 
    tokens=list(set(palabras_dos + entidades_dos))
)
print(f"Cantidad de tópicos antes de la fusión: {len(modelo_combinado.get_topics())}")
modelo_combinado = BERTopic.merge_models([modelo_combinado, modelo_dos], min_similarity=0.85)
guardar_documentos_a_db(repositorio, df_dia_dos, modelo_combinado, analizador_sentimientos, dia_tres)
guardar_modelo_a_db(repositorio, modelo_combinado)

# EDA de modelos fusionados para los días 0, 1 y 2
print(len(modelo_cero.get_topics()))
print(len(modelo_uno.get_topics()))
print(len(modelo_dos.get_topics()))
print(len(modelo_combinado.get_topics()))

info_topicos_2 = modelo_dos.get_topic_info()
info_topicos_combinados = modelo_combinado.get_topic_info()

print(info_topicos_2.tail(5))
print(info_topicos_combinados.tail(10))

nombre_topico_modelo_dos = "49_toneladas_exportaciones_producción_maíz"
resumenes_seleccionados = [item[0] for item in zip(textos_dos, topicos_dos) if item[1] == info_topicos_2.loc[info_topicos_2["Name"] == nombre_topico_modelo_dos, "Topic"].values[0]]
print(resumenes_seleccionados)
print(modelo_dos.transform(resumenes_seleccionados))
print(modelo_combinado.transform(resumenes_seleccionados))


# %%

# CASO 4: ANALIZAR UN DÍA COMPLETO DE NOTICIAS (DÍA 0, DÍA 1, DÍA 2 y DÍA 3)

dia_cuatro = "2024-07-12"

df_dia_tres, textos_tres, palabras_tres, entidades_tres = cargar_datos(dataset=ds_list[3])
topicos_tres, probabilidades_tres, modelo_tres = generar_topicos_del_dia(
    textos=textos_tres, 
    tokens=list(set(palabras_tres + entidades_tres))
)
print(f"Cantidad de tópicos antes de la fusión: {len(modelo_combinado.get_topics())}")
modelo_combinado = BERTopic.merge_models([modelo_combinado, modelo_tres], min_similarity=0.85)
guardar_documentos_a_db(repositorio, df_dia_tres, modelo_combinado, analizador_sentimientos, dia_cuatro)
guardar_modelo_a_db(repositorio, modelo_combinado)

# EDA de modelos fusionados para los días 0, 1, 2 y 3
print(len(modelo_cero.get_topics()))
print(len(modelo_uno.get_topics()))
print(len(modelo_dos.get_topics()))
print(len(modelo_tres.get_topics()))
print(len(modelo_combinado.get_topics()))

info_topicos_2 = modelo_tres.get_topic_info()
info_topicos_combinados = modelo_combinado.get_topic_info()

print(info_topicos_2.tail(5))
print(info_topicos_combinados.tail(10))

nombre_topico_modelo_tres = ""  # Actualmente está vacío

if nombre_topico_modelo_tres and nombre_topico_modelo_tres in info_topicos_2["Name"].values:
    valores_filtrados = info_topicos_2.loc[info_topicos_2["Name"] == nombre_topico_modelo_tres, "Topic"].values
    resumenes_seleccionados = [item[0] for item in zip(textos_tres, topicos_tres) if item[1] == valores_filtrados[0]]
    
    print(resumenes_seleccionados)
    print(modelo_tres.transform(resumenes_seleccionados))
    print(modelo_combinado.transform(resumenes_seleccionados))
else:
    print(f"No se encontró el nombre del tópico '{nombre_topico_modelo_tres}' en info_topicos_2['Name'].")
    resumenes_seleccionados = []



# %%
# ANALISIS DE TOPICOS EN EL TIEMPO

import pandas as pd

# Fechas de inicio y fin para el análisis
fecha_inicio = dia_inicial
fecha_fin = dia_tres

docs, timestamps = buscar_documentos(repositorio.get_client(), fecha_inicio, fecha_fin, cantidad=4500)

documentos_df = pd.DataFrame(
    {
        "Documento": docs,
        "ID": range(len(docs)),
        "Topico": modelo_combinado.topics_,
        "Imagen": None
    }
)
documentos_por_topico = documentos_df.groupby(['Topico'], as_index=False).agg({'Documento': ' '.join})

modelo_combinado.vectorizer_model = modelo_cero.vectorizer_model

c_tf_idf, _ = modelo_combinado._c_tf_idf(documentos_por_topico)
modelo_combinado.c_tf_idf_ = c_tf_idf

modelo_combinado.transform(docs)

topicos_en_tiempo = modelo_combinado.topics_over_time(
    docs=docs, 
    timestamps=timestamps
)
modelo_combinado.visualize_topics_over_time(topicos_en_tiempo)
