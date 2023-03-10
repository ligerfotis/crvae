### Prerequisites

    pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
    pip install opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64
    pip install -r requirements.txt
Install https://github.com/Quasimondo/RasterFairy.git

### Train

    python3 train/train_AE_custom.py -d MNIST -ls 128 -bs 256 -e 10 -lr 1e-3 -nw 16 -sh -aug --architecture custom 


### Evaluate
Local Model

    python3 evaluate/model_evaluator.py -d MNIST -n MNIST_AE_custom_l128_s0 -rp MNIST_AE_custom_l128_s0.ckp -nw 16 -bs 256

Wandb Model

    python3 evaluate/model_evaluator.py -d MNIST -n MNIST_AE_custom_l128_s0 -rp <you-github-name>/autoencoders/<run_id> -nw 16 -bs 256 --wandb