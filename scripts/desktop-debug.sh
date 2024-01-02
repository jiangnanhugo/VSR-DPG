#!/usr/bin/zsh
basepath=/home/jiangnan/PycharmProjects/scibench//
py3=/home/jiangnan/miniconda3/bin/python
set -x
for pgn in $basepath/data/unencrypted/equations_trigometric/sincos_nv3_nt22_prog_1.in; do
	#        echo "submit $pgn"r -p $dump_dir

	for bsl in DSR; do
		$py3 $basepath/cvDSO/main.py $basepath/cvDSO/config/config_regression_${bsl}.json --equation_name $pgn --noise_type normal --noise_scale 0.0
	done
done
