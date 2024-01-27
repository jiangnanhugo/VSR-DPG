#!/usr/bin/zsh
basepath=/home/$USER/PycharmProjects/cvdso
py3=/home/$USER/miniconda3/bin/python

type=$1
nv=$2
nt=$3
set -x
noise_type=normal
noise_scale=0.0
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
	$py3 $basepath/E2E/main.py --equation_name $datapath/$eq_name \
		--pretrained_model_filepath $basepath/E2E/model.pt --mode cuda\
		--noise_type $noise_type --noise_scale $noise_scale #>$dump_dir/prog_${prog}.noise_${noise_type}${noise_scale}.opt$opt.${bsl}.cvdso.out
done
