#! /bin/bash

#SBATCH --partition=ali
#SBATCH --account=alicpt
#SBATCH --qos=regular
#SBATCH --job-name=cmb_0_alicpt
#SBATCH --nodes=8
#SBATCH --exclude=aliws[005-020,026,028]
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
export PATH=/sharefs/alicpt/users/zrzhang/codes/checkProj/utils:$PATH

uuid=$(uuidgen)
echo $uuid
targetDir=TEST/1
mkdir -p tmp/$uuid
mkdir -p $targetDir

# generate lensed cmb
genLensedMap 1024 tmp/$uuid
cp -r tmp/$uuid $targetDir/LensInfo

# smoothing 
smoothing tmp/$uuid/CMB_LEN.fits tmp/$uuid

# add foreground
add tmp/$uuid/Ali_95.fits ~/workdir/PSM/checkDeproj/FG_beamsys/observations/Simulation_uKCMB/group1_map_95GHz.fits tmp/$uuid/tmp_ali_95.fits
add tmp/$uuid/Ali_150.fits ~/workdir/PSM/checkDeproj/FG_beamsys/observations/Simulation_uKCMB/group1_map_150GHz.fits tmp/$uuid/tmp_ali_150.fits

add tmp/$uuid/WMAP_K.fits ~/workdir/PSM/checkDeproj/FG_beamsys/observations/Simulation_uKCMB/group1_map_23GHz.fits $targetDir/WMAP_K.fits
add tmp/$uuid/HFI_100.fits ~/workdir/PSM/checkDeproj/FG_beamsys/observations/Simulation_uKCMB/group1_map_100GHz.fits $targetDir/HFI_100.fits
add tmp/$uuid/HFI_143.fits ~/workdir/PSM/checkDeproj/FG_beamsys/observations/Simulation_uKCMB/group1_map_143GHz.fits $targetDir/HFI_143.fits
add tmp/$uuid/HFI_217.fits ~/workdir/PSM/checkDeproj/FG_beamsys/observations/Simulation_uKCMB/group1_map_217GHz.fits $targetDir/HFI_217.fits
add tmp/$uuid/HFI_353.fits ~/workdir/PSM/checkDeproj/FG_beamsys/observations/Simulation_uKCMB/group1_map_353GHz.fits $targetDir/HFI_353.fits

# prepare template
genTemplate tmp/$uuid/tmp_ali_95.fits 19 19 tmp/$uuid/A95
genTemplate tmp/$uuid/tmp_ali_150.fits 11 11 tmp/$uuid/A150

realization ~/workdir/codes/checkProj/Datas/HFI_NOISE/SIGMA_HFI_100.fits tmp/$uuid/n100.fits
realization ~/workdir/codes/checkProj/Datas/HFI_NOISE/SIGMA_HFI_143.fits tmp/$uuid/n143.fits

add $targetDir/HFI_100.fits tmp/$uuid/n100.fits tmp/$uuid/tmp_hfi_100.fits
add $targetDir/HFI_143.fits tmp/$uuid/n143.fits tmp/$uuid/tmp_hfi_143.fits

genTemplate tmp/$uuid/tmp_hfi_100.fits 9.68200 19 tmp/$uuid/P95
genTemplate tmp/$uuid/tmp_hfi_143.fits 7.30300 11 tmp/$uuid/P150

mpirun -np 8 python -u ./deproj/OBS_beamsys.py tmp/$uuid
python solve.py tmp/$uuid $targetDir

# mv tmp/$uuid/*.fits $targetDir/
