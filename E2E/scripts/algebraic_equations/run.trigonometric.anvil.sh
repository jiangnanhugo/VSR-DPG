#!/usr/bin/zsh

basepath=/anvil/projects/x-cis230379/jiang631/data/cvdso
py310=/home/x-jiang631/workspace/miniconda3/envs/py310/bin/python3
type=$1
nv=$2
nt=$3

thispath=$basepath/E2E
datapath=$basepath/data/algebraic_equations/equations_trigonometric
opt=L-BFGS-B

noise_type=normal
noise_scale=0.0
metric_name=neg_mse
for prog in {0..9}; do
	eq_name=${type}_nv${nv}_nt${nt}_prog_${prog}.in
	echo "submit $eq_name"
	dump_dir=$basepath/result/${type}_nv${nv}_nt${nt}/$(date +%F)
	if [ ! -d "$dump_dir" ]; then
		echo "create dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/$(date +%F)
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	echo "$dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.out"
	sbatch -A cis230379 --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="E2E${type}${nv}${nt}"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.opt${opt}.e2e.out
#SBATCH --constraint=A
#SBATCH --time=24:00:00


hostname
$py310 $thispath/main.py --equation_name $datapath/$eq_name \
		--pretrained_model_filepath $basepath/E2E/model.pt \
		--mode cpu \
		--noise_type normal --noise_scale 0.0 >$dump_dir/prog_${prog}.noise_${noise_type}${noise_scale}.e2e.out
EOT
done
