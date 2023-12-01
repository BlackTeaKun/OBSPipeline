#! /bin/bash

#SBATCH --partition=ali
#SBATCH --account=alicpt
#SBATCH --qos=regular
#SBATCH --job-name=cmb_0_alicpt
#SBATCH --nodes=8
#SBATCH --exclude=aliws[005-020]
#SBATCH --cpus-per-task=64
#SBATCH -n 8
#SBATCH --mem-per-cpu=512M
#SBATCH --output=/sharefs/alicpt/users/zrzhang/codes/checkProj/sys.out


#=================================
#===== Run your job commands =====
#=================================
#source /afs/ihep.ac.cn/users/l/lisy/.bashrc
source /afs/ihep.ac.cn/users/z/zrzhang/.bashrc
micromamba activate cosmo

# uuid=$(uuidgen)
# mkdir -p tmp/$uuid
mpirun -np 8 python3 -u newOBS.py
# python solve.py tmp/$uuid TEST
