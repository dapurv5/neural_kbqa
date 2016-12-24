#!/usr/bin/python

HOPS_FROM_QN_ENTITY = 2
HOPS_FROM_RELEVANT_ENTITY = 1

MAX_CANDIDATE_ENTITIES = 128
MAX_CANDIDATE_PATHS = 1024
CLIP_CANDIDATE_PATHS_BETWEEN_SINGLE_PAIR = True

USE_RELEVANT_ENTITIES = True
USE_NBR_QN_ENTITIES = True #keep this true always otherwise you might run into no paths found
USE_NBR_RELEVANT_ENTITIES = True
#USE_NBR_ANSWER_ENTITIES

REMOVE_HIGH_DEGREE_ANSWER_ENTITIES = False #what was the release date of the film almighty thor

MAX_PATH_LENGTH = 3 #this includes the source and target entities, 3 translates to 1 intermediate node