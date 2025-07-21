# setup
pip install -v -e .
wandb login
ln -s /data/dataset/nuscenes_real ./data/nuscenes

# prepare
tar -xf nuScenes-lidarseg-mini-v1.0.tar.bz2
/opt/conda/envs/py38/bin/python ./tools/create_data_fusionocc.py
/opt/conda/envs/py38/bin/python ./img_seg/gen_segmap.py ./data/nuscenes --parallel=32

# test
/opt/conda/envs/py38/bin/python ./tools/test.py configs/fusion_occ/fusion_occ.py /data/models/fusion_occ_mask.pth 

# train
/opt/conda/envs/py38/bin/python ./tools/train.py ./configs/fusion_occ/fusion_occ.py
