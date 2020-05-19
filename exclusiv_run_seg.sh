#!/bin/bash 

##load modules
module load Anaconda3/2018.12 
module load GCC/8.3.0
module load CUDA/10.1.243
module load OpenMPI/3.1.4	
module load TensorFlow/2.1.0-Python-3.7.4
module load iccifort/2019.5.281
module load impi/2018.5.288
module load matplotlib/3.1.1-Python-3.7.4

##execute python file
python3 pg_reuse_more_exp_7.py  
