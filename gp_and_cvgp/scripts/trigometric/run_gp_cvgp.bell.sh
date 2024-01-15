#!/bin/bash -l

basepath=/depot/yexiang/apps/jiang631/data/scibench
py310=/home/jiang631/workspace/miniconda3/envs/py310/bin/python3
type=$1
nv=$2
nt=$3

thispath=$basepath/gp_and_vsr_gp
data_path=$basepath/data/unencrypted/equations_trigometric

noise_type=normal
noise_scale=0.0
metric_name=neg_mse
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
	sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="GP${type}${nv}${nt}_${prog}"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.gp.out
#SBATCH --constraint=A
#SBATCH --time=12:00:00
#SBATCH --mem=4GB

hostname

$py310 $thispath/try_gp_xyx.py --equation_name $data_path/$eq_name \
        		--metric_name $metric_name --noise_type $noise_type --noise_scale $noise_scale \
        		 > $dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.gp.out

EOT

	sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="VSR-GP${type}${nv}${nt}_${prog}"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.cvgp.out
#SBATCH --constraint=A
#SBATCH --time=12:00:00
#SBATCH --mem=4GB

hostname

$py310 $thispath/try_gp_xyx.py --equation_name $data_path/$eq_name --expand_gp \
        		--metric_name $metric_name --noise_type $noise_type --noise_scale $noise_scale \
        		 > $dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.cvgp.out

EOT

done
