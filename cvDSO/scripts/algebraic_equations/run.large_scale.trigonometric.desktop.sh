#!/usr/bin/zsh
basepath=/depot/yexiang/apps/jiang631/data/cvdso
py3=/home/jiang631/workspace/miniconda3/envs/py310/bin/python
type=$1
totalvars=$2
datapath=$basepath/data/algebraic_equations/large_scale_${totalvars}
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0
metric_name=inv_nrmse
n_cores=8
for prog in {0..9}; do
	for rand in {0..9}; do
		eq_name=${type}_nv8_nt812_prog_${prog}_totalvars_${totalvars}_rand_$rand.in
		echo "submit $eq_name"

		dump_dir=$basepath/result/${type}_nv8_nt812/totalvars_${totalvars}/$(date +%F)
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
		for bsl in DSR; do
			echo $basepath/cvDSO/config/config_regression_${bsl}.json
			sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=$n_cores <<EOT
#!/bin/bash -l

#SBATCH --job-name="cvDSO-${type}${totalvars}_${prog}_${rand}"
#SBATCH --output=$log_dir/${eq_name}.noise_${noise_type}_${noise_scale}.${bsl}.cvdso.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00

hostname

$py3 $basepath/cvDSO/main.py $basepath/cvDSO/config/config_regression_${bsl}.json --equation_name $datapath/$eq_name \
--optimizer $opt --metric_name $metric_name --n_cores $n_cores --noise_type $noise_type --noise_scale $noise_scale  >  $dump_dir/prog_${prog}.rand${rand}.noise_${noise_type}${noise_scale}.opt$opt.${bsl}.cvdso.out

EOT
		done
	done
done
