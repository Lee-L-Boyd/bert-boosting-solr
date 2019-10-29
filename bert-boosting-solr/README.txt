Teaching computers to answer questions using only English text.  
The problem I address in this project is whether the Bidirectional Encoder Representations for Transformers
(BERT) can be scaled efficiently for information retrieval.  
Project employed offline feature extraction of the documents, combined with a single layer of attention (neural networks) with query representations.  The model is trained to boost token values for the IR task of finding relevant documents for a question.   
The model had 1% increase in accuracy vs the optimized BM25 baseline in Solr.

The data is converted from the initial Squad dataset in create_pickles directory.
The model is trained in the hybrid-BM25-connect directory.
Solr queries are automated for both the baseline and the boosted model in solr-8.1.1 directory.  The data is also moved to Solr using the files within this directory.
The bert_edits file is just a couple minor changes that were done to bert to extract a little bit of additional information.
Note: The bert files are not included in this repository overall.  They are available at the Bert for Pytorch repository:
https://github.com/huggingface/transformers

The data from Squad is mostly emitted do to it's size, though the squad_paras.xml is a generated config file of the data before it is put into the Solr database.
The file structure is slightly different than it was on the system this was run on, so there might be some configurations to be done on directory paths to get it running.
