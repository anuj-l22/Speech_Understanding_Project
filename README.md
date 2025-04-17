# CSL7770 : Speech Understanding ( By Anuj Rajan Lalla [B22AI061])

  
This README conatins instructions on how to run the code for the details, check out the report 


If you want to work on a subset of data go to the data_extraction folder and execute this command

```bash
python extractor.py
```

This will create two folders : train folder and test folder


train folder will contain 800 samples
test folder will conatin 200 samples

To watermark these samples using AudioSeal use the command in the data_extraction folder itself

 ```bash
python audioseal_watermarkinge.py 
```

Move the watermarked folders (train and test) outside the data_extractor folder and then run this command

```bash
python defense_pipeline.py --mode train  --gpu 0
```

Also move the watermarked test files into this path: black-box/audiomarkdata_audioseal_max_5s/watermarked_200


This will generate a .pth file , you have to move this .pth file into the blackbox folder 


[.pth files is also provided on this drive link](https://drive.google.com/file/d/1wKo9uB3cH_F3T3AJcj9n_AoHQeiQkjJ7/view?usp=sharing)

Go into the blackbox folder and uncomment the line for the attack that you want to run in run.sh

For eg:

If  you want to run ALIF before applying adversarial defense uncomment this command in run.sh

```bash
python3 alif_attack.py --gpu 3 --testset_size 200 --query_budget 10 --tau 0.15 --model audioseal --blackbox_folder alif_10kbefore --attack_type both --eps 0.05
```

And if you want to run after loading the defense weights uncomment this

```bash
python3 alif_attack.py --gpu 3 --testset_size 200 --query_budget 10 --tau 0.15 --model audioseal --blackbox_folder alif_10kafter --attack_type both --eps 0.05  --def_ok True
```
