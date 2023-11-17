#!/usr/bin/zsh
basepath=/home/jiangnan/PycharmProjects/scibench
py3615=/home/jiangnan/anaconda3/envs/py3615/bin/python3

for pgn in /home/jiangnan/PycharmProjects/scibench/data/unencrypted/equations_feynman/FeynmanICh43Eq31.in; do
	#        echo "submit $pgn"
	dump_dir=$basepath/result/$(date +%F)
	if [ ! -d "$dump_dir" ]; then
		echo "create dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	for bsl in DSR PQT VPG GPMELD; do
		echo $bsl, $(date +'%R/%m/%d/%Y')
		$py3615 -m dso.run $basepath/dso_classic/config/config_regression_${bsl}.json --equation_name $pgn --noise_type normal --noise_scale 0.0 #>$dump_dir/data.${bsl}.out
	done
done
