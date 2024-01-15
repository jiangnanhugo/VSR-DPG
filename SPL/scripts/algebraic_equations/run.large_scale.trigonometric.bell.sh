#!/usr/bin/zsh
basepath=/depot/yexiang/apps/jiang631/data/cvdso
py3=/home/jiang631/workspace/miniconda3/envs/py310/bin/python
type=$1
totalvars=$2
data_path=$basepath/data/algebraic_equations/large_scale_${totalvars}
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0
metric_name=neg_mse
num_episodes=10000
for prog in {0..9}; do
	#prog=0
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

		sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="SPL${type}${totalvars}_${prog}_${rand}"
#SBATCH --output=$log_dir/${eq_name}.noise_${noise_type}_${noise_scale}.spl.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00
#SBATCH --mem=8GB
hostname

$py3 $basepath/SPL/main.py --equation_name $data_path/$eq_name --optimizer $opt  \
	--num_episodes $num_episodes \
	--metric_name $metric_name --noise_type $noise_type --noise_scale $noise_scale > $dump_dir/prog_${prog}_rand_$rand.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.spl.out
EOT
	done
done
