## Prepare model files

Please download the human model files and place them in the following structure:

```
|-- smpl
|   |-- SMPL_NEUTRAL.pkl
|   |-- SMPL_MALE.pkl
|   |-- SMPL_FEMALE.pkl
|-- smplx
|   |-- MANO_SMPLX_vertex_ids.pkl
|   |-- SMPL-X__FLAME_vertex_ids.npy
|   |-- SMPLX_NEUTRAL.pkl
|   |-- SMPLX_to_J14.pkl
|   |-- SMPLX_NEUTRAL.npz
|   |-- SMPLX_MALE.npz
|   |-- SMPLX_FEMALE.npz
|-- mano
|   |-- MANO_LEFT.pkl
|   |-- MANO_RIGHT.pkl
|-- flame
|   |-- flame_dynamic_embedding.npy
|   |-- flame_static_embedding.pkl
|   |-- FLAME_NEUTRAL.pkl
```



### **NOTE**:

The hosting servers may require you to create an account and explicitly agree with the license before downloading the data or models. Since different files are hosted on different servers, you may need to sign up for multiple accounts and log in multiple times. This process can be tedious, please be patient. 

Directly accessing the download link without logging in may result in a “Download denied” error. for example, using wget directly to download file like below can't download the file

```
wget --no-check-certificate https://psfiles.is.tuebingen.mpg.de/downloads/mano/mano_v1_2-zip
```

The best way to download the file remains to use your browser

Links provided may automatically redirect to the login page. If this happens, please log in and try again.


### Instructions

Here we provide the download links for each file:

- **smpl**:

  - [SMPL_NEUTRAL.pkl](https://github.com/sampepose/smpl_models/raw/master/SMPL_NEUTRAL.pkl)
  - [SMPL_MALE.pkl and SMPL_FEMALE.pkl](https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip)
    - An account is required to download. Register and log in to download `SMPL_python_v.1.0.0.zip`.
    - Unzip the`SMPL_python_v.1.0.0.zip`
      - Rename `basicModel_f_lbs_10_207_0_v1.0.0.pkl` to `SMPL_FEMALE.pkl`.
      - Rename `basicModel_m_lbs_10_207_0_v1.0.0.pkl` to `SMPL_MALE.pkl`.

- **smplx**:

  - [MANO_SMPLX_vertex_ids.pkl and SMPL-X__FLAME_vertex_ids.npy](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_mano_flame_correspondences.zip)
  - [SMPLX_NEUTRAL.pkl,SMPLX_NEUTRAL.npz, SMPLX_MALE.npz, SMPLX_FEMALE.npz](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip)
  - SMPLX_to_J14.pkl
    - Log in to [EXPOSE](https://expose.is.tue.mpg.de/download.php) first.
    - Click `ExPose Model (477 MB)` to download [expose_data.zip](https://download.is.tue.mpg.de/download.php?domain=expose&resume=1&sfile=expose_data.zip), which contains `SMPLX_to_J14.pkl`.
    - You can also follow this [repo](https://github.com/vchoutas/expose#preparing-the-data) to prepare the data `SMPLX_to_J14.pkl`.
  
- **mano**:
  
    - Log in to [MANO](https://mano.is.tue.mpg.de/download.php) first.
    - Click on `Models & Code` to download [mano_v1_2-zip](https://psfiles.is.tuebingen.mpg.de/downloads/mano/mano_v1_2-zip), which contains `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`.
    
- **flame**:

  - `flame_dynamic_embedding.npy` and `flame_static_embedding.pkl`

    - ```
      git clone https://github.com/soubhiksanyal/RingNet
      ```

    - You can find the files in the `RingNet/flame_model/` directory.

    - If you are working on Linux, the `flame_static_embedding.pkl` file may have a pickle load issue. Check the fix below.

  - [FLAME_NEUTRAL.pkl](https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=FLAME2019.zip)

    - An account is required to download. Register and log in to download `FLAME2019.zip`.
    - Rename `generic_model.pkl` to `FLAME_NEUTUTRAL.pkl`.



#### Other

If you encounter a pickle load issue with the `flame_static_embedding.pkl` file, the error may look like this:

```
grounded-sam-osx/utils/smplx/smplx/body_models.py:1860 in __init__                     
│   1857 │   │                                                                         
│   1858 │   │   with open(landmark_bcoord_filename, 'rb') as fp:                       
│   1859 │   │   │   print(f"landmark_bcoord_filename:{landmark_bcoord_filename}")     
│ ❱ 1860 │   │   │   landmarks_data = pickle.load(fp, encoding='latin1')               
│   1861 │   │                                                                         
│   1862 │   │   lmk_faces_idx = landmarks_data['lmk_face_idx'].astype(np.int64)       
│   1863 │   │   self.register_buffer('lmk_faces_idx', 
```



Use the following script to fix the issue:

```
import pickle

def try_read(file_name):
    # Error may occur:
    # _pickle.UnpicklingError: the STRING opcode argument must be quoted
    with open(file_name, 'rb') as f:
        data = pickle.load(f,  encoding='latin1')
    print(data)

# To fix the error caused by Linux and Windows end-of-line differences
def rewrite(original, destination):
    content = ''
    with open(original, 'rb') as infile:
        content = infile.read()
    with open(destination, 'wb') as output:
        for line in content.splitlines():
            output.write(line + str.encode('\n'))

if __name__ == "__main__":
    file_name = "flame_static_embedding.pkl"
    rewrite(file_name, file_name)
    try_read(file_name)
```



This script rewrites the file with the correct end-of-line characters and attempts to read the file again.
