#!/usr/bin/zsh
basepath=/home/jiangnan/PycharmProjects/cvdso
py3=/home/jiangnan/miniconda3/bin/python

type=$1
nv=$2
nt=$3
set -x
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0
metric_name=inv_nrmse
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

	for bsl in DSR; do
		$py3 $basepath/cvDSO/main.py $basepath/cvDSO/config/config_regression_${bsl}.json --equation_name $datapath/$eq_name \
		--noise_type normal --noise_scale 0.0 #>$dump_dir/prog_${prog}.noise_${noise_type}${noise_scale}.opt$opt.${bsl}.cvdso.out
	done
done
