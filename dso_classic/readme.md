
## 2. Run DSR, PQT, VPG, GPMeld

### Our Modification
We add our dataoracle and change the dataloader from reading a large ".csv" file to a active query API.

### 2.0 prequisites

1. install python environment 3.6.13: `conda create -n py3613 python=3.6.13`.
2. use the enviorment `conda env py3613`.
3. install `dso`

```cmd
cd ./dso
pip install --upgrade setuptools pip
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS"
pip install -e ./dso
```

3. create the `.csv` data file and `.json` model configuration file


```bash
# generate the **Noiseless** **[inv, sincos, sincosinv]** dataset with configurations *(5,5,8)*.
./dso/dataset/gen_data.sh
# generate the **Noisy** **[inv, sincos, sincosinv]** dataset with configurations *(5,5,8)*.
./dso/dataset/noisy_gen_data.sh
```

4. run DSR, PQT, VPG, GPMeld models by
   If you want to run DSR, PQT, VPG, GPMeld on **Noiseless** datasets.

```bash
./dso/scripts/run_dsr_pqt_vpg_gpmeld.sh
```

If you want to run DSR, PQT, VPG, GPMeld on **Noisy** datasets.

```bash
./dso/scripts/noisy_run_dsr_pqt_vpg_gpmeld.sh
```