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

