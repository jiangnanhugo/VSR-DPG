#!/usr/bin/zsh
basepath=/home/jiangnan/PycharmProjects/scibench//
py3=/home/jiangnan/miniconda3/bin/python
set -x
for pgn in 0 1 2 3 4 5 6 7 8 9; do
	filename=$basepath/data/unencrypted/equations_trigometric/sincos_nv3_nt22_prog_$pgn.in
	#        echo "submit $pgn"r -p $dump_dir

	for bsl in DSR; do
		$py3 $basepath/cvDSO/main.py $basepath/cvDSO/config/config_regression_${bsl}.json --equation_name $filename --noise_type normal --noise_scale 0.0 > sincos_nv3_nt22_prog_$pgn.log
	done
done
