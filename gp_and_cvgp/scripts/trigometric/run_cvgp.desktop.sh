#!/usr/bin/zsh



basepath=/home/jiangnan/PycharmProjects/scibench/
py3=/home/jiangnan/miniconda3/bin/python

type=$1
nv=$2
nt=$3

thispath=$basepath/gp_and_vsr_gp
data_path=$basepath/data/unencrypted/equations_trigometric

noise_type=normal
noise_scale=0.0
metric_name=neg_mse
opt=BFGS
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
	echo "$dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.optim_$opt.cvgp.out"
	$py3 $thispath/main.py --equation_name $data_path/$eq_name --optimizer $opt --cvgp \
		--metric_name $metric_name --noise_type $noise_type --noise_scale $noise_scale >$dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.optim_$opt.cvgp.out &
done
