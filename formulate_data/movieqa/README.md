Knowledge Base Question Answering Experiments & Code
-------

### Dataset Preparation
```
python clean_entities.py --input $HOME/Downloads/qa_datasets/movieqa/knowledge_source/entities.txt --output ../../data/movieqa/clean_entities.txt



python clean_kb.py --input_kb $HOME/Downloads/qa_datasets/movieqa/knowledge_source/full/full_kb.txt --input_entities ../../data/movieqa/clean_entities.txt --output_graph ../../data/movieqa/clean_full_kb_graph.txt  --output_doc ../../data/movieqa/clean_full_kb_doc.txt
python clean_kb.py --input_kb $HOME/Downloads/qa_datasets/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt --input_entities ../../data/movieqa/clean_entities.txt --output_graph ../../data/movieqa/clean_wiki-entities_kb_graph.txt  --output_doc ../../data/movieqa/clean_wiki-entities_kb_doc.txt



python clean_qa.py --input_examples $HOME/Downloads/qa_datasets/movieqa/questions/wiki_entities/wiki-entities_qa_train.txt --input_entities ../../data/movieqa/clean_entities.txt --output_examples ../../data/movieqa/clean_wiki-entities_qa_train.txt
python clean_qa.py --input_examples $HOME/Downloads/qa_datasets/movieqa/questions/wiki_entities/wiki-entities_qa_test.txt --input_entities ../../data/movieqa/clean_entities.txt --output_examples ../../data/movieqa/clean_wiki-entities_qa_test.txt
python clean_qa.py --input_examples $HOME/Downloads/qa_datasets/movieqa/questions/wiki_entities/wiki-entities_qa_dev.txt --input_entities ../../data/movieqa/clean_entities.txt --output_examples ../../data/movieqa/clean_wiki-entities_qa_dev.txt

python clean_qa.py --input_examples $HOME/Downloads/qa_datasets/movieqa/questions/full/full_qa_train.txt --input_entities ../../data/movieqa/clean_entities.txt --output_examples ../../data/movieqa/clean_full_qa_train.txt
python clean_qa.py --input_examples $HOME/Downloads/qa_datasets/movieqa/questions/full/full_qa_test.txt --input_entities ../../data/movieqa/clean_entities.txt --output_examples ../../data/movieqa/clean_full_qa_test.txt
python clean_qa.py --input_examples $HOME/Downloads/qa_datasets/movieqa/questions/full/full_qa_dev.txt --input_entities ../../data/movieqa/clean_entities.txt --output_examples ../../data/movieqa/clean_full_qa_dev.txt



python gen_stopwords.py --input_examples ../../data/movieqa/clean_wiki-entities_qa_train.txt  --output ../../data/movieqa/stopwords.txt
python gen_dictionaries.py



```

### LICENSE
Copyright (c) 2017 Apurv Verma