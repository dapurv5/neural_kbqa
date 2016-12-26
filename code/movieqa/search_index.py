#!/usr/bin/python

import codecs
import unicodedata

from whoosh import qparser
from whoosh import scoring
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.filedb.filestore import RamStorage

from text_util import clean_line
from clean_utils import read_file_as_dict


PIPE = "|"
SPACE = " "

class SearchIndex(object):
  def __init__(self, doc_path, stopwords=None):
    st = RamStorage()
    st.create()
    schema = Schema(entity_name=TEXT(stored=True), fieldname=TEXT(stored=True), content=TEXT())
    self.ix = st.create_index(schema)
    writer = self.ix.writer()
    self.remove_stopwords_while_indexing = False
    if stopwords:
      self.remove_stopwords_while_indexing = True
      self.stopwords_dict = read_file_as_dict(stopwords)

    with codecs.open(doc_path, 'r', "utf-8") as doc_file:
      for line in doc_file:
        line = clean_line(line)
        entity_name, fieldname, content = line.split(PIPE)
        writer.add_document(entity_name=entity_name, fieldname=fieldname, content=self.remove_stopwords_from_text(content))
    writer.commit()

  def remove_stopwords_from_text(self, content):
    words = content.split(SPACE)
    words_clean = []
    for word in words:
      if self.remove_stopwords_while_indexing and word not in self.stopwords_dict:
        words_clean.append(word)
    return " ".join(words_clean) if len(words_clean) > 0 else content

  def get_candidate_docs(self, question, limit=20):
    docs = set([])
    question = self.remove_stopwords_from_text(question)
    with self.ix.searcher() as searcher:
      query = QueryParser("content", self.ix.schema, group=qparser.OrGroup).parse(question)
      results = searcher.search(query, limit=limit)
      for result in results:
        docs.add(result['entity_name'])
    #TODO: remove the unicode normalization since this will be done earlier in the pipeline
    docs = [unicodedata.normalize('NFKD', doc).encode('ascii','ignore') for doc in docs]
    return docs

if __name__=="__main__":
  searcher = SearchIndex("../../data/movieqa/clean_wiki-entities_kb_doc.txt",
                         "../../data/movieqa/stopwords.txt")
  print searcher.get_candidate_docs("ginger rogers and")
