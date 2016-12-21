#!/usr/bin/python

import argparse
import csv
import networkx as nx

from text_util import clean_line

PIPE = "|"
HIGH_DEGREE_THRESHOLD = 50

class KnowledgeGraph(object):
  def __init__(self, graph_path, unidirectional=True):
    """
    directed: a value of false indicates that for every fact (e1, R, e2) inserted in the KB,
              (e2, invR, e1) will be inserted in the KB
    """
    self.G = nx.DiGraph()
    with open(graph_path, 'r') as graph_file:
      for line in graph_file:
        line = clean_line(line)
        e1, relation, e2 = line.split(PIPE)
        self.G.add_edge(e1, e2, {"relation": relation})
        if not unidirectional:
          self.G.add_edge(e2, e1, {"relation": self.get_inverse_relation(relation)})

    self.high_degree_nodes = set([])
    indeg = self.G.in_degree()
    for v in indeg:
      if indeg[v] > HIGH_DEGREE_THRESHOLD:
        self.high_degree_nodes.add(v)


  def get_inverse_relation(self, relation):
    return "INV_"+relation

  def get_all_paths(self, source, target, cutoff):
    """
    Returns two lists, the first is a list of paths from src to target where each path is itself a list
    The paths are represented by nodes (entities) on the path.
    The second list is a list of paths from src to target where each path is itself a list
    The paths are represented by the edge types (relations) on the path
    [ [e1, e2], [e1, e3, e2]], [[r1], [r2, r3] ]
    cutoff: represents the max length of the path allowed
    """
    paths_of_entities = []
    paths_of_relations = []
    for path in nx.all_simple_paths(self.G, source, target, cutoff):
      paths_of_entities.append(path)
      relations_path = []
      for i in range(0, len(path)-1):
        relation = self.G[path[i]][path[i+1]]['relation']
        relations_path.append(relation)
      paths_of_relations.append(relations_path)
    return paths_of_entities, paths_of_relations

  def get_candidate_neighbors(self, node, num_hops=2, avoid_high_degree_nodes=True):
    """
    Get all the n hops neighbors from a node in the graph
    avoid_high_degree_nodes = True, skips any path going through a high degree node
    See constructor for definition of high degree nodes
    """
    result = set([])
    q = [node]
    visited = set([node])
    dist = {node: 0}
    while len(q) > 0:
      u = q.pop(0)
      result.add(u)
      for nbr in self.G.neighbors(u):
        if nbr in self.high_degree_nodes and avoid_high_degree_nodes:
          continue
        if nbr not in visited:
          visited.add(nbr)
          dist[nbr] = dist[u] + 1
          if dist[nbr] <= num_hops:
            q.append(nbr)
    result.remove(node)
    return result

  def log_statistics(self):
    print "NUM_NODES", len(nx.nodes(self.G))


if __name__ == "__main__":
  graph_path = "../../data/movieqa/clean_wiki-entities_kb_graph.txt"
  kb = KnowledgeGraph(graph_path, unidirectional=False)
  entities_paths, relations_paths = kb.get_all_paths(source="moonraker", target="lewis gilbert", cutoff=3)
  print kb.get_candidate_neighbors("moonraker")
  print len(kb.get_candidate_neighbors("moonraker", num_hops=2))
