#!/usr/bin/zsh

basepath=/depot/yexiang/apps/jiang631/data/scibench
py37=/home/jiang631/workspace/miniconda3/envs/py37/bin/python
type=$1
nv=$2
nt=$3

thispath=$basepath/dso_classic
datapath=$basepath/data/unencrypted/equations_trigometric
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0
#metric_name=neg_nmse

for prog in {0..9}; do
	eq_name=${type}_nv${nv}_nt${nt}_prog_${prog}.in
	echo "submit $eq_name"

	dump_dir=$basepath/result/${type}_nv${nv}_nt${nt}/$(date +%F)
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/$(date +%F)
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	for bsl in DSR PQT VPG GPMELD; do
		echo $basepath/dso_classic/config/config_regression_${bsl}.json
		echo $datapath/$eq_name
		echo $dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.opt${opt}.${bsl}
		sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=8 <<EOT
#!/bin/bash -l
#SBATCH --job-name="$bsl-${type}${nv}${nt}${prog}"
#SBATCH --output=$log_dir/${eq_name}.noise_${noise_type}_${noise_scale}.${bsl}.out
#SBATCH --constraint=A
#SBATCH --time=12:00:00
#SBATCH --mem=4GB

hostname

$py37 $thispath/run.py $basepath/dso_classic/config/config_regression_${bsl}.json --equation_name $datapath/$eq_name \
--logdir $dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.opt${opt}.${bsl} \
--noise_type $noise_type --noise_scale $noise_scale > $dump_dir/prog_${prog}.noise_${noise_type}${noise_scale}.opt$opt.${bsl}.out

EOT
	done
done

#done
