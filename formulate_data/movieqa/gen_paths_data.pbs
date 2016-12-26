#!/bin/bash
#PBS -N gen_paths_data_job
#PBS -q monkeys
#PBS -l walltime=36:00:00
#PBS -l nodes=2:ppn=8
#PBS -l mem=16gb
#PBS -k oe
#PBS -m ae
#PBS -M apurvverma@gatech.edu

cd $PBS_O_WORKDIR
#echo "
#==================================================
#Node: `hostname`
#Working directory: ${PWD}
#==================================================
#"

module unload gcc
module load gcc/4.9.0
module load cudnn/7.5


export PYTHONPATH="/nv/hcoc1/averma80/ProgramFiles/tensorflow_gpu":${PYTHONPATH}
/usr/local/pacerepov1/python/2.7/gcc-4.9.0/bin/python gen_paths_data.py --input_examples ../../data/movieqa/clean_wiki-entities_qa_train.txt --input_graph ../../data/movieqa/clean_wiki-entities_kb_graph.txt --input_doc ../../data/movieqa/clean_wiki-entities_kb_doc.txt --output_examples ../../data/movieqa/wiki-entities_train_paths.txt --stopwords ../../data/movieqa/stopwords.txt --mode train