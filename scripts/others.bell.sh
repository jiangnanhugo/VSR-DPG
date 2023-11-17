#!/usr/bin/zsh
#basepath=/home/jiangnan/PycharmProjects/scibench
basepath=/depot/yexiang/apps/jiang631/data/scibench
#py3615=/home/jiangnan/anaconda3/envs/py3615/bin/python3
py3615=/home/jiang631/workspace/miniconda3/envs/py3615/bin/python3

for pgn in Constant_1.in Keijzer_1.in Livermore_10.in Livermore2_Vars2_17.in Livermore2_Vars3_15.in Livermore2_Vars4_13.in Livermore2_Vars5_11.in Livermore2_Vars5_9.in Livermore2_Vars6_7.in Livermore2_Vars7_5.in Neat_8.in R1a.in Constant_2.in Keijzer_2.in Livermore_11.in Livermore2_Vars2_18.in Livermore2_Vars3_16.in Livermore2_Vars4_14.in Livermore2_Vars5_12.in Livermore2_Vars6_10.in Livermore2_Vars6_8.in Livermore2_Vars7_6.in Neat_9.in R1.in Constant_3.in Keijzer_3.in Livermore_12.in Livermore2_Vars2_19.in Livermore2_Vars3_17.in Livermore2_Vars4_15.in Livermore2_Vars5_13.in Livermore2_Vars6_11.in Livermore2_Vars6_9.in Livermore2_Vars7_7.in Nguyen_10.in R2a.in Constant_4.in Keijzer_4.in Livermore_13.in Livermore2_Vars2_1.in Livermore2_Vars3_18.in Livermore2_Vars4_16.in Livermore2_Vars5_14.in Livermore2_Vars6_12.in Livermore2_Vars7_10.in Livermore2_Vars7_8.in Nguyen_11.in R2.in Constant_5.in Keijzer_5.in Livermore_14.in Livermore2_Vars2_20.in Livermore2_Vars3_19.in Livermore2_Vars4_17.in Livermore2_Vars5_15.in Livermore2_Vars6_13.in Livermore2_Vars7_11.in Livermore2_Vars7_9.in Nguyen_12a.in R3a.in Constant_6.in Keijzer_6.in Livermore_15.in Livermore2_Vars2_21.in Livermore2_Vars3_1.in Livermore2_Vars4_18.in Livermore2_Vars5_16.in Livermore2_Vars6_14.in Livermore2_Vars7_12.in Livermore_3.in Nguyen_12.in R3.in Constant_7.in Keijzer_7.in Livermore_16.in Livermore2_Vars2_22.in Livermore2_Vars3_20.in Livermore2_Vars4_19.in Livermore2_Vars5_17.in Livermore2_Vars6_15.in Livermore2_Vars7_13.in Livermore_4.in Nguyen_1.in Sine.in Constant_8.in Keijzer_8.in Livermore_17.in Livermore2_Vars2_23.in Livermore2_Vars3_21.in Livermore2_Vars4_1.in Livermore2_Vars5_18.in Livermore2_Vars6_16.in Livermore2_Vars7_14.in Livermore_5.in Nguyen_2.in Vladislavleva_1.in Const_Test_1.in Keijzer_9.in Livermore_18.in Livermore2_Vars2_24.in Livermore2_Vars3_22.in Livermore2_Vars4_20.in Livermore2_Vars5_19.in Livermore2_Vars6_17.in Livermore2_Vars7_15.in Livermore_6.in Nguyen_3.in Vladislavleva_2.in Const_Test_2.in Korns_10.in Livermore_19.in Livermore2_Vars2_25.in Livermore2_Vars3_23.in Livermore2_Vars4_21.in Livermore2_Vars5_1.in Livermore2_Vars6_18.in Livermore2_Vars7_16.in Livermore_7a.in Nguyen_4.in Vladislavleva_3.in GrammarVAE_1.in Korns_11.in Livermore_1.in Livermore2_Vars2_2.in Livermore2_Vars3_24.in Livermore2_Vars4_22.in Livermore2_Vars5_20.in Livermore2_Vars6_19.in Livermore2_Vars7_17.in Livermore_7.in Nguyen_5.in Vladislavleva_4.in Jin_1.in Korns_12.in Livermore_20.in Livermore2_Vars2_3.in Livermore2_Vars3_25.in Livermore2_Vars4_23.in Livermore2_Vars5_21.in Livermore2_Vars6_1.in Livermore2_Vars7_18.in Livermore_8a.in Nguyen_6.in Vladislavleva_5.in Jin_2.in Korns_1.in Livermore_21.in Livermore2_Vars2_4.in Livermore2_Vars3_2.in Livermore2_Vars4_24.in Livermore2_Vars5_22.in Livermore2_Vars6_20.in Livermore2_Vars7_19.in Livermore_8.in Nguyen_7.in Vladislavleva_6.in Jin_3.in Korns_2.in Livermore_22.in Livermore2_Vars2_5.in Livermore2_Vars3_3.in Livermore2_Vars4_25.in Livermore2_Vars5_23.in Livermore2_Vars6_21.in Livermore2_Vars7_1.in Livermore_9.in Nguyen_8.in Vladislavleva_7.in Jin_4.in Korns_3.in Livermore_23.in Livermore2_Vars2_6.in Livermore2_Vars3_4.in Livermore2_Vars4_2.in Livermore2_Vars5_24.in Livermore2_Vars6_22.in Livermore2_Vars7_20.in Meier_3.in Nguyen_9.in Vladislavleva_8.in Jin_5.in Korns_4.in Livermore_2.in Livermore2_Vars2_7.in Livermore2_Vars3_5.in Livermore2_Vars4_3.in Livermore2_Vars5_25.in Livermore2_Vars6_23.in Livermore2_Vars7_21.in Meier_4.in Nonic.in Jin_6.in Korns_5.in Livermore2_Vars2_10.in Livermore2_Vars2_8.in Livermore2_Vars3_6.in Livermore2_Vars4_4.in Livermore2_Vars5_2.in Livermore2_Vars6_24.in Livermore2_Vars7_22.in Neat_1.in Pagie_1.in Keijzer_10.in Korns_6.in Livermore2_Vars2_11.in Livermore2_Vars2_9.in Livermore2_Vars3_7.in Livermore2_Vars4_5.in Livermore2_Vars5_3.in Livermore2_Vars6_25.in Livermore2_Vars7_23.in Neat_2.in Poly_10.in Keijzer_11.in Korns_7.in Livermore2_Vars2_12.in Livermore2_Vars3_10.in Livermore2_Vars3_8.in Livermore2_Vars4_6.in Livermore2_Vars5_4.in Livermore2_Vars6_2.in Livermore2_Vars7_24.in Neat_3.in Poly_1.in Keijzer_12.in Korns_8.in Livermore2_Vars2_13.in Livermore2_Vars3_11.in Livermore2_Vars3_9.in Livermore2_Vars4_7.in Livermore2_Vars5_5.in Livermore2_Vars6_3.in Livermore2_Vars7_25.in Neat_4.in Poly_2.in Keijzer_13.in Korns_9.in Livermore2_Vars2_14.in Livermore2_Vars3_12.in Livermore2_Vars4_10.in Livermore2_Vars4_8.in Livermore2_Vars5_6.in Livermore2_Vars6_4.in Livermore2_Vars7_2.in Neat_5.in Poly_3.in Keijzer_14.in Koza_2.in Livermore2_Vars2_15.in Livermore2_Vars3_13.in Livermore2_Vars4_11.in Livermore2_Vars4_9.in Livermore2_Vars5_7.in Livermore2_Vars6_5.in Livermore2_Vars7_3.in Neat_6.in Poly_4.in Keijzer_15.in Koza_3.in Livermore2_Vars2_16.in Livermore2_Vars3_14.in Livermore2_Vars4_12.in Livermore2_Vars5_10.in Livermore2_Vars5_8.in Livermore2_Vars6_6.in Livermore2_Vars7_4.in Neat_7.in Poly_5.in; do
	trimed_name=${pgn:-3}
	echo "submit $trimed_name"
	equation_name=$basepath/data/unencrypted/equations_others/$pgn
	dump_dir=$basepath/result/Others/$(date +%F)
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/Others/$(date +%F)
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	for bsl in DSR PQT VPG GPMELD; do
		echo $bsl, $(date +'%R/%m/%d/%Y')
		sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=8 <<EOT
#!/bin/bash -l

#SBATCH --job-name="$bsl-${trimed_name}"
#SBATCH --output=$log_dir/run_${bsl}_${pgn}.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00
#SBATCH --mem=4096MB

hostname

$py3615 -m dso.run $basepath/dso_classic/config/config_regression_${bsl}.json --equation_name $equation_name --noise_type normal --noise_scale 0.0  > $dump_dir/$pgn.${bsl}.out

EOT
	done
done

#done
