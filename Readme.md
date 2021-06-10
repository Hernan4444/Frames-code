# Código Frames

Este repositorio incluye:

- Código para preprocesar dataset y crear los embeddings (Fasttext, Elmo, Beto CLS, Beto Average). (**Code - Create final dataset and embeddings.ipynb**)
- Códigos usados para entrenar cada modelo (SVM, MLP, Regresión Logistica, RF, LSTM, Beto & Baselines) (**Code - Clasificador_\*.ipynb **)
- Código para obtener embeddings por folds de cada texto. Incluye embeddins preentrenados (Fasttext, Elmo, Beto CLS, Beto Average) y de modelos de deep learning (LSTM & Beto). (**Code - Split embedding per fold.ipynb**)
- Código para cargar los embedding de train y test según el fold indicado. (**Code - Load Data & Embedding Frame.ipyn**)

**Importante** la carpeta dataset debe ser llenada con los archivos `fold_{i}_train.npy` y `fold_{i}_test.npy` que están en el drive compartido vía mail (privado). Por otro lado, el código importante a utilizar es **Code - Load Data & Embedding Frame.ipynb**, en ese notebook se muestra como cargar un DataFrame con los textos separados por fold, por conjunto (train, test) y, para cada texto, están los posibles embeddings. Las columnas relevantes son:

- `original_text`: es el texto original de la noticia.
- `preprocess_text`: es el texto preprocesado de la noticia utilizando el vocabulario presente en `vocabulary_corpus_counter.txt`.
- `encoded`: es el texto preprocesado de la noticia utilizando el vocabulario presente en `vocabulary_corpus_counter.txt`, pero cada token fue transformado a un ID único por token.
- `conflicto, economico, humanidad, moral`: columnas que indican la presencia de dicho frames en la noticia. Corresponde al LABEL a predecir.
- `fasttext`: representación del texto usando fasttext por token y calculando su promedio.
- `elmo`: representación del texto usando Elmo por token y calculando su promedio.
- `beto_embedding_mean`: representación del texto usando Beto por token y calculando su promedio.
- `beto_embedding_cls`: representación del texto usando fasttext por token y utilizando solo el vector del primer token que corresponde al CLS.
- `Bi-LSTM`: representación del texto tras pasarlo por un Bi-LSTM. Este modelo fue entrenado con la función de pérdida cross-entropy.
- `Bi-LSTM_AsymetricLoss`: representación del texto tras pasarlo por un Bi-LSTM. Este modelo fue entrenado con la función de pérdida [Asymetric Loss](https://arxiv.org/abs/2009.14119).
- `Beto-finetunning_cross_entropy`: representación del texto tras pasarlo por un modelo Beto cuya última capa fue finetunieada para la tarea de detección de frames. Este modelo fue entrenado con la función de pérdida cross-entropy.
- `Beto-finetunning_asymetric`: representación del texto tras pasarlo por un modelo Beto cuya última capa fue finetunieada para la tarea de detección de frames. Este modelo fue entrenado con la función de pérdida [Asymetric Loss](https://arxiv.org/abs/2009.14119).
- `tf-idf`: representación del texto usando TF-IDF.

## Archivos faltantes

Por temas de espacio, faltan 2 archivos:
- embeddings-l-model.vec
- pytorch_model.bin

Ambos están en el siguiente drive: https://drive.google.com/drive/folders/1zBbw68oKVyhlcqu9Ds9zd4EL0tYWHsld