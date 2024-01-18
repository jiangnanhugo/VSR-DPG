#!/usr/bin/zsh

basepath=/home/$USER/data/cvdso
py37=/home/$USER/workspace/miniconda3/envs/py37/bin/python


type=$1
totalvars=$2
datapath=$basepath/data/algebraic_equations/large_scale_${totalvars}
thispath=$basepath/dso_classic
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0

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
	for bsl in DSR PQT VPG GPMELD; do
		echo $basepath/dso_classic/config/config_regression_${bsl}.json
		echo $datapath/$eq_name
		echo $dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.opt${opt}.${bsl}
		sbatch -A cis230379 --nodes=1 --ntasks=1 --cpus-per-task=8 <<EOT
#!/bin/bash -l
#SBATCH --job-name="$bsl-T${totalvars}_R${rand}"
#SBATCH --output=$log_dir/${eq_name}.noise_${noise_type}_${noise_scale}.${bsl}.out
#SBATCH --constraint=A
#SBATCH --time=12:00:00
#SBATCH --mem=4GB

hostname

$py37 $thispath/run.py $basepath/dso_classic/config/config_regression_${bsl}.json --equation_name $datapath/$eq_name \
--logdir $dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.opt${opt}.${bsl} \
--noise_type $noise_type --noise_scale $noise_scale > $dump_dir/prog_${prog}_rand_$rand.noise_${noise_type}${noise_scale}.opt$opt.${bsl}.out

EOT

	done
done
