

```python
"""
Using tensorflow to generate word embeddings and using nearest neighbour alorightm to find sentance similarity
"""
import tensorflow as tf
import tensorflow_hub as hub
import pickle
from database import Database

embedding_module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
# This might take a while if running for the first time as it downloads the module and caches it for later use.
embed = hub.Module(embedding_module_url)
```

    INFO:tensorflow:Using /var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules to cache modules.
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_0:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_0
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_1:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_1
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_10:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_10
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_11:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_11
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_12:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_12
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_13:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_13
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_14:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_14
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_15:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_15
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_16:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_16
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_2:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_2
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_3:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_3
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_4:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_4
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_5:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_5
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_6:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_6
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_7:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_7
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_8:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_8
    INFO:tensorflow:Initialize variable module/Embeddings_en/sharded_9:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Embeddings_en/sharded_9
    INFO:tensorflow:Initialize variable module/Encoder_en/DNN/ResidualHidden_0/weights:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Encoder_en/DNN/ResidualHidden_0/weights
    INFO:tensorflow:Initialize variable module/Encoder_en/DNN/ResidualHidden_1/weights:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Encoder_en/DNN/ResidualHidden_1/weights
    INFO:tensorflow:Initialize variable module/Encoder_en/DNN/ResidualHidden_2/weights:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Encoder_en/DNN/ResidualHidden_2/weights
    INFO:tensorflow:Initialize variable module/Encoder_en/DNN/ResidualHidden_3/projection:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Encoder_en/DNN/ResidualHidden_3/projection
    INFO:tensorflow:Initialize variable module/Encoder_en/DNN/ResidualHidden_3/weights:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with Encoder_en/DNN/ResidualHidden_3/weights
    INFO:tensorflow:Initialize variable module/SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_0/bias:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_0/bias
    INFO:tensorflow:Initialize variable module/SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_0/weights:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_0/weights
    INFO:tensorflow:Initialize variable module/SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_1/bias:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_1/bias
    INFO:tensorflow:Initialize variable module/SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_1/weights:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_1/weights
    INFO:tensorflow:Initialize variable module/SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_2/bias:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_2/bias
    INFO:tensorflow:Initialize variable module/SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_2/weights:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SHARED_RANK_ANSWER/response_encoder_0/tanh_layer_2/weights
    INFO:tensorflow:Initialize variable module/SNLI/Classifier/LinearLayer/bias:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SNLI/Classifier/LinearLayer/bias
    INFO:tensorflow:Initialize variable module/SNLI/Classifier/LinearLayer/weights:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SNLI/Classifier/LinearLayer/weights
    INFO:tensorflow:Initialize variable module/SNLI/Classifier/tanh_layer_0/bias:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SNLI/Classifier/tanh_layer_0/bias
    INFO:tensorflow:Initialize variable module/SNLI/Classifier/tanh_layer_0/weights:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with SNLI/Classifier/tanh_layer_0/weights
    INFO:tensorflow:Initialize variable module/global_step:0 from checkpoint b'/var/folders/kx/_79d_8t10wb9tw0c996plxxm0000gn/T/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/variables/variables' with global_step



```python
# Loading event data for model evaluation
event_data = Database.get_instance().list_companies_by_event('ijl_18')
event_data = [str(c['summary'])
              .strip()
              .lower()
              .replace('\r', '')
              .replace('\n', '')
              for c in event_data if str(c['summary']).lower() != 'none']
```


```python
# Generating word embeddings for loaded event data
def embedder(session_, input_tensor_, messages_, encoding_tensor):
    embeddings = session_.run(
        encoding_tensor, feed_dict={input_tensor_: messages_}
    )
    return embeddings

similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    embeddings = embedder(session, similarity_input_placeholder, event_data,
                          similarity_message_encodings)
    with open('event_embeddings.bin', 'wb') as fl:
        pickle.dump(embeddings, fl)
```


```python
from sklearn.neighbors import NearestNeighbors
X = pickle.load(open('event_embeddings.bin', 'rb'))
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

# Relevant to the trained model context
message = [
    "diamond jewellery"
]
similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    y = embedder(session, similarity_input_placeholder, message,
                 similarity_message_encodings)
    distances, indices = nbrs.kneighbors(y, 5)
    for i in range(len(indices[0])):
        print('Company Description: ', event_data[indices[0][i]])
```

    Company Description:  envi jewellery limited - manufacturer of 18k and platinum diamond and semi precious stone jewellery, specialising in making unique design jewellery to meet individual needs.
    Company Description:  silver designer jewellery with gemstonesgold designer jewellery with gemstones
    Company Description:  ro jewellery designed by karolis ro černeckis. it’s made of silver and gold using precious and semi-precious stones. each unique design passes through the hands of skilled jeweller – that is what makes this jewellery so elegant and special. simple yet timeless – it’s ro jewellery.
    Company Description:  fine jewellery in 18 kt. gold with diamonds and precious stones all made in italy
    Company Description:  our product range comprises of all kinds of jewellery items, such as bracelets, necklaces, rings, earrings, pendants, all of which are in plain gold or with cz, in 9,10,14 and 18 carat gold or 925 silver



```python
from sklearn.neighbors import NearestNeighbors
X = pickle.load(open('event_embeddings.bin', 'rb'))
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

# Irrelevant to the trained model context
message = [
    "microsoft windows"
]
similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    y = embedder(session, similarity_input_placeholder, message,
                 similarity_message_encodings)
    distances, indices = nbrs.kneighbors(y, 5)
    for i in range(len(indices[0])):
        print('Company Description: ', event_data[indices[0][i]])
```

    Company Description:  everybody at jewelmaster is committed to providing the best software to support your business. we listen to you to make sure that every change we make meets your needs. we update our software as often as is practical to keep you on top.
    Company Description:  freeform fabrication ltd a supplier of technology tools & services to jewellery designers & manufacturersmain products in our portfolio:	solidscape 3d printers	laser welders	laser engravers/markers	solutionix 3d scannersgeography: united kingdom, ireland, belgium, netherlands, luxemburga professional team to provide on-site tech
    Company Description:  with nearly 30 years of experience in developing artistic cad/cam software type3 presents 3design and 3shaper for the jewellery/accessory industry and typeedit and lasertype for the engraving industry.
    Company Description:  uk based watch distribution company - supplying on-line and bricks & mortar retailers in the uk, caribbean & alaska.
    Company Description:  a pioneer of ecommerce, we help retailers grow their business online. how? through our unrivalled platform coupled with our online marketing and support.



```python
"""
I couldn't find any recommended method to evaluate the performance of semantic similarity. 
What's different here in comparison traditional way of finding simantic similarity is robust word embeddings.
We can take it for granted that this method provides somewhat accurate semantic sentence similairty just because
the embeddings are generated using USE - Universal Sentence Encoding model is trained and optimized for 
greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of 
data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language 
understanding tasks.
"""
```
