#!/usr/bin/zsh
basepath=/home/$USER/data/cvdso
py3=/home/$USER/miniconda3/envs/py310/bin/python3.10
#
type=Lorenz
datapath=$basepath/data/differential_equations/
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0
metric_name=neg_mse
num_episodes=10000
set -x
for prog in {0..2}; do
	eq_name=${type}_d${prog}.in
	echo "submit $eq_name"

	dump_dir=$basepath/result/${type}/$(date +%F)
	echo $dump_dir
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	nohup $py3 $basepath/SPL/main.py --equation_name $datapath/$eq_name --optimizer $opt \
		--num_episodes $num_episodes \
		--metric_name $metric_name --noise_type $noise_type --noise_scale $noise_scale >$dump_dir/${type}_d${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.spl.out &
done
