conda create -n rw-env python=3.9.16 -y 
conda activate rw-env        
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg  
pip uninstall numpy -y 
pip install numpy==1.23.5    