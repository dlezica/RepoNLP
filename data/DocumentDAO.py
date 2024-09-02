from opensearchpy import Field, Document, Text, Date

TOPIC_DIMENSIONS = 384
knn_params = {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "nmslib"
}
data_index = 'data-index'

class KNNVector(Field):
    name = "knn_vector"
    def __init__(self, dimension, method, **kwargs):
        super(KNNVector, self).__init__(dimension=dimension, method=method, **kwargs)

class News(Document):
    id = KNNVector(TOPIC_DIMENSIONS, knn_params)
    created_at = Date()
    text = Text()
    entities = Text()
    sentiment_analysis = Text()

    class Index:
        name = data_index

    def save(self, ** kwargs):
        return super(News, self).save(** kwargs)