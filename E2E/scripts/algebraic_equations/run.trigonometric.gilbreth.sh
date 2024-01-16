#!/usr/bin/zsh

basepath=/home/$USER/data/cvdso
py3=/home/$USER/workspace/miniconda3/envs/py310/bin/python
type=$1
nv=$2
nt=$3
set -x
noise_type=normal
noise_scale=0.0
datapath=$basepath/data/algebraic_equations/equations_trigonometric
set -x
for prog in {0..9};
do
	eq_name=${type}_nv${nv}_nt${nt}_prog_${prog}.in
	echo "submit $eq_name"

	dump_dir=$basepath/result/${type}_nv${nv}_nt${nt}/$(date +%F)

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

#SBATCH --job-name="E2E-${type}_nv${nv}_nt${nt}"
#SBATCH --output=$log_dir/${eq_name}.noise_${noise_type}_${noise_scale}.e2e.out
#SBATCH --time=3:59:00

hostname

$py3 $basepath/E2E/main.py --equation_name $datapath/$eq_name \
		--pretrained_model_filepath $basepath/E2E/model.pt --mode cuda \
		--noise_type normal --noise_scale 0.0 >$dump_dir/prog_${prog}.noise_${noise_type}${noise_scale}.e2e.out

EOT
done
