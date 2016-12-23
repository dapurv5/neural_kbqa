#!/usr/bin/python

import codecs
import unicodedata

from whoosh import qparser
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser

from text_util import clean_line


PIPE = "|"

class SearchIndex(object):
  def __init__(self, doc_path):
    schema = Schema(entity_name=TEXT(stored=True), fieldname=TEXT(stored=True), content=TEXT())
    self.ix = create_in("/tmp", schema)
    writer = self.ix.writer()

    with codecs.open(doc_path, 'r', "utf-8") as doc_file:
      for line in doc_file:
        line = clean_line(line)
        entity_name, fieldname, content = line.split(PIPE)
        writer.add_document(entity_name=entity_name, fieldname=fieldname, content=content)
    writer.commit()

  def get_candidate_docs(self, question):
    docs = set([])
    with self.ix.searcher() as searcher:
      query = QueryParser("content", self.ix.schema, group=qparser.OrGroup).parse(question)
      results = searcher.search(query)
      for result in results:
        docs.add(result['entity_name'])
    #TODO: remove the unicode normalization since this will be done earlier in the pipeline
    docs = [unicodedata.normalize('NFKD', doc).encode('ascii','ignore') for doc in docs]
    return docs

if __name__=="__main__":
  searcher = SearchIndex("../../data/movieqa/clean_wiki-entities_kb_doc.txt")
  print searcher.get_candidate_docs("ginger rogers and")
