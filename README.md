### Prerequisites

    pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
    pip install opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64
    pip install -r requirements.txt

### Train

    python3 train/main.py -d MNIST --z-dim 16 -bs 256 -e 50 -si 10 -lr 1e-3 -nw 8 -sh --beta 1 --gamma 1 --seed 101 --loss mse --opt sgd

