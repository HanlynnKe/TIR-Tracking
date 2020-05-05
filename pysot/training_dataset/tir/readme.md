# Preprocessing TIR(Thermal infrared videos dataset)
Thermal InfraRed Dataset build for [MMNet](https://github.com/QiaoLiuHit/MMNet)

### Download dataset (14GB)

Download the dataset from [Baidu NetDisk](https://pan.baidu.com/s/1uLQ8pHsAbBq8hRFM2aOhYg) or [MEGA Drive](https://mega.nz/#F!BtAVyAwb!hRsl9pCKp5JmJur_UHpEwA)

````shell
cat * > tir.zip
unzip tir.zip

cd TIRDataset-MMNet
mkdir Annotations Annotations/TIR Annotations/TIR/train
mkdir Annotations/TIR/train/a Annotations/TIR/train/b Annotations/TIR/train/c
mkdir Data Data/TIR Data/TIR/train
mkdir Data/TIR/train/a Data/TIR/train/b Data/TIR/train/c
cd ..
````

### Crop & Generate data info (10 min)

````shell
python split_tir.py
python parse_tir.py

#python par_crop.py [crop_size] [num_threads]
python par_crop.py 511 12
python gen_json.py
````
