#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=16000
##SBATCH --ntasks-per-node=1
##SBATCH --exclusive
#SBATCH --job-name="enseble"
#SBATCH --output=test-srun.out
#SBATCH --mail-user=torresrl@stud.ntnu.no
#SBATCH --mail-type=ALL


echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"#SBATCH --mail-user=torresrl
#SBATCH --mail-type=ALL


# load modules
module load Anaconda3/2018.12 
module load GCC/8.3.0
module load CUDA/10.1.243
module load OpenMPI/3.1.4	
module load TensorFlow/2.1.0-Python-3.7.4
module load iccifort/2019.5.281
module load impi/2018.5.288
module load matplotlib/3.1.1-Python-3.7.4

python3 pg_reuse_more_enseble_exp_7.py 
