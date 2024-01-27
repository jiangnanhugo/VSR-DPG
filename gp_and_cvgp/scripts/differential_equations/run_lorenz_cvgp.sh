basepath=/home/$USER/PycharmProjects/scibench

outputdir=$basepath/result/lorenz/$(date +%F)
if [ ! -d "$outputdir" ]; then
	echo "create dir: $outputdir"
	mkdir -p $outputdir
fi
equation_name=$basepath/data/unencrypted/equations_pde/Lorenz_dx.in
python $basepath/gp_and_cvgp/try_gp_xyx.py --equation_name $equation_name --metric_name neg_nmse --noise_type normal --noise_scale 0.0 --cvgp>$outputdir/lorenz_dx.cvgp.out &

equation_name=$basepath/data/unencrypted/equations_pde/Lorenz_dy.in
python $basepath/gp_and_cvgp/try_gp_xyx.py --equation_name $equation_name --metric_name neg_nmse --noise_type normal --noise_scale 0.0 --cvgp>$outputdir/lorenz_dy.cvgp.out &

equation_name=$basepath/data/unencrypted/equations_pde/Lorenz_dz.in
python $basepath/gp_and_cvgp/try_gp_xyx.py --equation_name $equation_name --metric_name neg_nmse --noise_type normal --noise_scale 0.0 --cvgp>$outputdir/lorenz_dz.cvgp.out &
