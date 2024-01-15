#!/usr/bin/zsh
basepath=/home/jiangnan/PycharmProjects/scibench
py37=/home/jiangnan/anaconda3/envs/py37/bin/python3.7

for pgn in /home/jiangnan/PycharmProjects/scibench/data/unencrypted/equations_trigometric/inv_nv2_nt11_prog_7.in; do

	for bsl in GPMELD; do
		echo $bsl, $(date +'%R/%m/%d/%Y')
		$py37 /home/jiangnan/PycharmProjects/scibench/dso_classic/run.py /home/jiangnan/PycharmProjects/scibench/dso_classic/config/config_regression_PQT.json \
		--equation_name /home/jiangnan/PycharmProjects/scibench/data/unencrypted/equations_trigometric/inv_nv2_nt11_prog_7.in --noise_type normal --noise_scale 0.0 #>$dump_dir/data.${bsl}.out
	done
done
