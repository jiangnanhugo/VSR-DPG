#!/usr/bin/zsh
basepath=/home/$USER/data/cvdso
py3=/home/$USER/workspace/miniconda3/envs/py310/bin/python
type=$1
totalvars=$2
datapath=$basepath/data/algebraic_equations/large_scale_${totalvars}

noise_type=normal
noise_scale=0.0
#for prog in {0..9}; do
prog=0
for rand in {0..9}; do
	eq_name=${type}_nv5_nt55_prog_${prog}_totalvars_${totalvars}_rand_$rand.in
	echo "submit $eq_name"

	dump_dir=$basepath/result/${type}_nv5_nt55/totalvars_${totalvars}/$(date +%F)
	echo $dump_dir
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/$(date +%F)
	echo $log_dir
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	sbatch -A standby --nodes=1 --ntasks=1 --gpus=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="E2E_T${totalvars}_R${rand}"
#SBATCH --output=$log_dir/${eq_name}.noise_${noise_type}_${noise_scale}.e2e.out
#SBATCH --time=3:59:00

hostname

$py3 $basepath/E2E/main.py --equation_name $datapath/$eq_name \
--pretrained_model_filepath $basepath/E2E/model.pt --mode cuda \
--noise_type $noise_type --noise_scale $noise_scale  >  $dump_dir/prog_${prog}_rand_$rand.noise_${noise_type}${noise_scale}.e2e.out

EOT
done
