# ASR project barebones

## Installation guide

Я делал git clone в датасфере или в каггле, устаналвивал requirements.txt, ставил key от wandb и запускал train. index я формировал локально, а зачем менял пути, чтобы они корректно были для каггл/датасферы. Индекс я клал рядом с датасетом и указывал save_dir для каггла, а для сферы я менял в коде index_path (там рядом нельзя было положить)
```shell
pip install -r ./requirements.txt
```
