#!/usr/bin/zsh
basepath=/depot/yexiang/apps/jiang631/data/cvDSO
py3=/home/jiang631/workspace/miniconda3/bin/python3
type=$1
nv=$2
nt=$3
datapath=$basepath/data/algebraic_equations/equations_trigonometric
set -x
for pgn in {0..9};
do
	eq_name=${type}_nv${nv}_nt${nt}_prog_${pgn}.in
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
	for bsl in DSR; do
		sbatch -A standby --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="cvDSO-${eq_name}"
#SBATCH --output=$log_dir/run_${bsl}_${pgn}.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00
#SBATCH --mem=4GB

hostname

$py3 $basepath/cvDSO/main.py $basepath/cvDSO/config/config_regression_${type}_${bsl}.json --equation_name $datapath/$eq_name --noise_type 'normal' --noise_scale 0.0  > $dump_dir/$pgn.${bsl}.cvdso.out

EOT
	done
done

#done
