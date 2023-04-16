Please down the human model files and place them here as the following structure:

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

Here we provide the download link of each files:

- **smpl**: [SMPL_NEUTRAL.pkl](https://github.com/sampepose/smpl_models/raw/master/SMPL_NEUTRAL.pkl), [SMPL_MALE.pkl and SMPL_FEMALE.pkl](https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip). You need to rename `basicModel_f_lbs_10_207_0_v1.0.0.pkl` and `basicModel_m_lbs_10_207_0_v1.0.0.pkl`  to `SMPL_FEMALE.pkl` and `SMPL_MALE.pkl`.

- **smplx**: [MANO_SMPLX_vertex_ids.pkl and SMPL-X__FLAME_vertex_ids.npy](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_mano_flame_correspondences.zip), [SMPLX_NEUTRAL.pkl, SMPLX_NEUTRAL.npz, SMPLX_MALE.npz, SMPLX_FEMALE.npz](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz), [SMPLX_to_J14.pkl](https://github.com/vchoutas/expose#preparing-the-data). 

- **mano**: [here](https://psfiles.is.tuebingen.mpg.de/downloads/mano/mano_v1_2-zip)

- **flame**:  [flame_dynamic_embedding.npy and flame_static_embedding.pkl](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model), [FLAME_NEUTRAL.pkl](https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=FLAME2019.zip). You need to rename `generic_model.pkl` to `FLAME_NEUTRAL.pkl`.