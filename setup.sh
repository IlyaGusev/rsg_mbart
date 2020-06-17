git clone https://github.com/NVIDIA/apex
rm apex/setup.py
wget https://gist.githubusercontent.com/IlyaGusev/d014117d80454e096b2d300a556817a6/raw/1aea06742150c4fa7a10c4ec4a63aa3cfe9b7fc1/setup.py -O apex/setup.py
cd apex && CUDA_HOME=/usr/local/cuda-10.2/ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./ && cd ..

git clone https://github.com/pytorch/fairseq
cd fairseq && git checkout 7a6519f84fed06947bbf161c7b66c9099bc4ce53 && pip install --editable . && cd ..

