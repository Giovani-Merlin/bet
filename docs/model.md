# Model

Full architecture can be seen on 'bet.text.model.model.py' file.
The model is basically any Transformer model (Using "AutoModel" from HuggingFace) with or without a ffn layer on top of it. The model is trained using a Bi-Encoder approach, where the query and the input data are encoded separately and then compared using cosine similarity using in-batch negatives.

The foward pass uses the CLS token and normalizes the output (to avoid giving more important to some candidates).

## Training

Full training class on 'bet.text.model.trainer.py' file.

Bi-encoder approach, that is, we have two base models being trained at the same time, one for the query and one for the input data.

We optimize for all the encoders layers (included the embedding one, as we need to train for the extra token "[ENT]" that is added to the input data) and for the "random_negatives_loss_parameter" - the conversion from the cosine similarity to logits.

The 'random_negatives_loss_parameter' has a totally diferent scale from the model's parameters. I did an adaptation to use the same optimizer (to use more than on optimizer/learning ratte we need to disable automatic optimization from pythorch lightning). That is, I get the mean value of the last layer weights and scale the random_negatives_loss_parameter to be in the same scale with the "re_escaler_factor". When training I use this scaled variable to be optimized but the real value is used when transforming the cosine similarity to logits.

### Loss

The loss is calculated using the cosine similarity between the query and the input data, using in-batch negatives. The implementation is similar as on BLINK, but I mask possible repeated candidates in the same batch to avoid wrong loss when the same candidate is repeated in the batch, also, I use cosine similarity instead of direct multiplication (as Google on Dense entity retrieval) to avoid giving more importance to popular cnadidates. Also, I keep track of the mean candidate similarity, to be used as a metric.

### Validation

Instead of using validation loss (dependent of the batch size, bigger batches are harder to predict as we have more candidates) I do directly the R@ metric on it. For that, I embedd ALL the candidates (one time each) and after having all the queries and candidates embedded I do the R@ metric. this approach gives morre accuracy and low impact on the training time if test/train are not so big - 5k for example is a small number.

The R@ metric is controlled by "testing_eval_recall" argument.
