#!/usr/bin/zsh
basepath=/home/jiangnan/PycharmProjects/scibench
py3615=/home/jiangnan/anaconda3/envs/py3615/bin/python3
set -x
for pgn in $basepath/data/unencrypted/equations_trigometric/sincos_nv3_nt22_prog_1.in; do
	#        echo "submit $pgn"
	dump_dir=$basepath/result/$(date +%F)
	if [ ! -d "$dump_dir" ]; then
		echo "create dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	for bsl in DSR; do
		echo $bsl, $(date +'%R/%m/%d/%Y')
		$py3615 $basepath/cvDSO/run.py $basepath/dso_classic/config/config_regression_${bsl}.json --equation_name $pgn --noise_type normal --noise_scale 0.0 #>$dump_dir/data.${bsl}.out
	done
done
