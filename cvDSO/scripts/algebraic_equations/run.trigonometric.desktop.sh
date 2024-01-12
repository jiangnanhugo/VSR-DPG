#!/usr/bin/zsh
basepath=/home/jiangnan/PycharmProjects/cvdso
py3=/home/jiangnan/miniconda3/bin/python

type=$1
totalvars=$2
datapath=$basepath/data/algebraic_equations/large_scale_${totalvars}
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0
metric_name=inv_nrmse
n_cores=4
set -x
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
		for bsl in DSR; do
			echo $basepath/cvDSO/config/config_regression_${bsl}.json
			$py3 $basepath/cvDSO/main.py $basepath/cvDSO/config/config_regression_${bsl}.json --equation_name $datapath/$eq_name \
				--optimizer $opt --metric_name $metric_name --n_cores $n_cores --noise_type $noise_type --noise_scale $noise_scale >$dump_dir/prog_${prog}.rand${rand}.noise_${noise_type}${noise_scale}.opt$opt.${bsl}.cvdso.out
		done
	done
done
