#!/usr/bin/zsh

basepath=/depot/yexiang/apps/jiang631/data/scibench
py37=/home/jiang631/workspace/miniconda3/envs/py37/bin/python
type=Livermore2
nv=$1

thispath=$basepath/dso_classic
data_path=$basepath/data/unencrypted/equations_livermore2


noise_type=normal
noise_scale=0.0
#metric_name=neg_nmse
for prog in {1..25}; do
	eq_name=${type}_Vars${nv}_$prog.in
	echo "submit $eq_name"

	dump_dir=$basepath/result/${type}_Vars${nv}/$(date +%F)
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
		echo $bsl, $(date +'%R/%m/%d/%Y')
		sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=8 <<EOT
#!/bin/bash -l

#SBATCH --job-name="$bsl-Vars${nv}_$prog"
#SBATCH --output=$log_dir/${eq_name}.noise_${noise_type}_${noise_scale}.${bsl}.out
#SBATCH --constraint=A
#SBATCH --time=10:00:00
#SBATCH --mem=4GB

hostname

$py37 $thispath/run.py $basepath/dso_classic/config/config_regression_${bsl}.json \
--equation_name $data_path/$eq_name --noise_type $noise_type --noise_scale $noise_scale \
 --logdir $dump_dir/${eq_name}.noise_${noise_type}${noise_scale}${bsl} > $dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.${bsl}.out


EOT
	done
done

#done
