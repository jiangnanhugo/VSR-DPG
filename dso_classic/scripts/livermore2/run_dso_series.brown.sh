#!/usr/bin/zsh
basepath=/depot/yexiang/apps/jiang631/data/scibench
py37=/home/jiang631/workspace/miniconda3/envs/py37/bin/python3.7
type=Livermore2
nv=$1
thispath=$basepath/dso_classic
datapath=$basepath/data/unencrypted/equations_livermore2


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
	for bsl in DSR PQT VPG; do
		echo $basepath/dso_classic/config/config_regression_${bsl}.json
		echo $datapath/$eq_name
		echo $dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.opt${opt}.${bsl}
		timeout 24h $py37 $thispath/run.py $basepath/dso_classic/config/config_regression_${bsl}.json --equation_name $datapath/$eq_name \
			--logdir $dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.${bsl} \
			--noise_type $noise_type --noise_scale $noise_scale >$dump_dir/prog_${prog}.noise_${noise_type}${noise_scale}.${bsl}.out
	done
done
