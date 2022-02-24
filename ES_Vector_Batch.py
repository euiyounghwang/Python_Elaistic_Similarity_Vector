
# https://skagh.tistory.com/32

import json
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Use tensorflow 1 behavior to match the Universal Sentence Encoder
# examples (https://tfhub.dev/google/universal-sentence-encoder/2).
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

"""
[QA]
StackOverFlow : https://github.com/elastic/rally-tracks/tree/master/so
https://github.com/jtibshirani/text-embeddings/tree/blog
https://blog.naver.com/olpaemi/221639136195
pigar -p ./requirements.txt -P ./ES_UnFair_Detection/
"""

##### INDEXING #####
def index_data():
    print("Creating the 'posts' index.")
    client.indices.delete(index=INDEX_NAME, ignore=[404])

    with open(INDEX_FILE) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=INDEX_NAME, body=source)

    docs = []
    count = 0

    with open(DATA_FILE) as data_file:
        for line in data_file:
            line = line.strip()

            doc = json.loads(line)
            # if doc["type"] != "question":
            #     continue

            docs.append(doc)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print("Indexed {} documents.".format(count))


        if docs:
            index_batch(docs)
            print("Indexed {} documents.".format(count))

    client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")


def index_batch(docs):
    titles = [doc["TITLE"] for doc in docs]
    title_vectors = embed_text(titles)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["title_vector"] = title_vectors[i]
        requests.append(request)
    bulk(client, requests)


##### SEARCHING #####
def run_query_loop():
    """

    :return:
    """

    handle_query()

    """
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return
    """


def handle_query():
    # query = input("Enter query: ")
    query = '견본주택 건립공사는 어떻게 진행되고 있는가?'
    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['title_vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["TITLE", "body"]}
        }
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()


##### EMBEDDING #####
def embed_text(text):

    print('\nembed_text -> {}'.format(text))

    vectors = session.run(embeddings, feed_dict={text_ph: text})
    return [vector.tolist() for vector in vectors]


##### MAIN SCRIPT #####
if __name__ == '__main__':
    INDEX_NAME = "posts"
    # INDEX_FILE = "INPUT/Research/index.json"
    INDEX_FILE = "INPUT/index.json"

    DATA_FILE = "INPUT/posts.json"
    BATCH_SIZE = 1000

    SEARCH_SIZE = 5

    print("Downloading pre-trained embeddings from tensorflow hub...")
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    text_ph = tf.placeholder(tf.string)
    embeddings = embed(text_ph)
    print("Done.")

    print("Creating tensorflow session...")
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print("Done.")

    client = Elasticsearch(hosts="10.132.17.117:9201", http_auth=('elastic', 'gsaadmin'),)

    # index_data()
    run_query_loop()

    print("Closing tensorflow session...")
    session.close()
    print("Done.")
