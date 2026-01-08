# Environment
Python 3.8    
PyTorch >= 1.5

```bash
git clone https://github.com/ljzycmd/SimDeblur.git

# install the SimDeblur
cd SimDeblur
bash Install.sh
```

Replace the path SimDeblur\simdeblur\dataset\dvd.py with [`dvd.py`](./dvd.py)  
In last_output_path = os.path.join("the path", video_name, "GT", "{}"), please replace "the path" with the input path of the first frame.  

Replace the path SimDeblur\simdeblur\dataset\gopro.py with [`gopro.py`](./gopro.py)  
In last_output_path = os.path.join("the path", video_name,"sharp","{}"), please replace "the path" with the input path of the first frame.  

Replace the path SimDeblur\simdeblur\dataset\bsd.py with [`bsd.py`](./bsd.py)  
In last_output_path = os.path.join("the path", video_name,"Sharp","RGB","{}"), please replace "the path" with the input path of the first frame.  

# Test
1. Download and unzip the datasets
2. Pretrained checkpoints are listed:
| Dataset | Link |
|---|---|
| DVD | [Link1](https://drive.google.com/file/d/10cRanHwJ-W9q5_EXaw075oXK3zkpEA0r/view?usp=drive_link) |
| GOPRO | Link2 |
| BSD 1ms-8ms | Link3 |
| BSD 3ms-24ms | Link4 |



