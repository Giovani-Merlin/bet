# Inference

For doing inference, we need to create a [SCANN](https://github.com/google-research/google-research/tree/master/scann) index with the candidates and then use the model to embed the queries and search for the nearest neighbors on the index. For now we have just 2 options of index, "brute_force":True and False. For true scann is doing just brute force approach, for False  we create a hash index with tree search. I tested some options and it seems to be the best compromise of speed, accuracy and dataset size. The code can be seen on "bet.text.vector_store.scann_ids.py" file.* The searcher handles the ids and the titles (if given) search.

We can create the index with model.create_index . After the index creation, we can search using model.search function, that returns the ids and the scores. The scores are the cosine similarity between the query and the candidate. Please note that queries and candidates must be on BET format.

* Also we can use "reorder:True", the accuracy increases just a bit but we need to keep the full embedded dataset - which can be really heavy when using full wikipedia. With reorder False we keep just the hashed dataset. But, if needs to increase the candidate pool time by time (adding new candidates) we need to keep the full dataset by using reorder True to avoid re-embedding the full dataset each time.
