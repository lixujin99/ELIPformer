# ELIPformer

The official implementation of **Integrating Language-Image Prior into EEG Decoding for Cross-Task Zero-Calibration RSVP-BCI**.

![alt text](figure/Model_framework.png)

The structure of the proposed ELIPformer. (a) ELIPformer consists of the feature extractor, the prompt encoder, the cross bi-attention module, and the fusion module. (b) The prompt encoder consists of components from the pre trained CLIP-ViT-B/32 [30]. The image encoder and text encoder are inherited from this model. Additionally, the patch embedding layer and transformer layers are derived from the image encoder in CLIP-ViT-B/32. (c) The cross bi-attention module is composed of L successive cross bi-attention layers for effective interaction between EEG features and language-image features.



## 1&nbsp; Installation

Follow the steps below to prepare the virtual environment.

Create and activate the environment:
```shell
conda create -n mtreenet python=3.10
conda activate ELIPformer
```

Install dependencies:
```shell
pip install -r requirements.txt
```


## 2&nbsp; Data

### 2.1&nbsp; Dataset

We designed and implemented three RSVP target image retrieval tasks to construct the “NeuBCI Target Retrieval RSVP-EEG Dataset” including EEG data and the corresponding stimulus images. Our collected dataset and corresponding data descriptions are released at [https://doi.org/10.57760/sciencedb.14812](https://doi.org/10.57760/sciencedb.14812).

Each participant's **electroencephalogram (EEG) data** are stored in `.npz` files, and **stimulus images** are stored in `.mat` files. Each EEG and image file with the same file name corresponds to the same participant and block. Below is the dataset structure:

| Task | Modality | File Format | File Naming | Data Shape | Label Description |
|------|----------|-------------|-------------|------------|-------------------|
| A    | EEG      | `.npz`       | `S1_1.npz` ~ `S43_5.npz` | `(trials × channels × time)` | `0`: non-target, `1`: target-1, `2`: target-2 |
| A    | EM       | `.mat`       | `S1_1.mat` ~ `S43_5.mat` | `(trials × time × components)` | `0`: non-target, `1`: target-1, `2`: target-2 |
| B    | EEG      | `.npz`       | `S1_1.npz` ~ `S43_5.npz` | `(trials × channels × time)` | `0`: non-target, `1`: target-1, `2`: target-2 |
| B    | EM       | `.mat`       | `S1_1.mat` ~ `S43_5.mat` | `(trials × time × components)` | `0`: non-target, `1`: target-1, `2`: target-2 |
| C    | EEG      | `.npz`       | `S1_1.npz` ~ `S43_5.npz` | `(trials × channels × time)` | `0`: non-target, `1`: target-1, `2`: target-2 |
| C    | EM       | `.mat`       | `S1_1.mat` ~ `S43_5.mat` | `(trials × time × components)` | `0`: non-target, `1`: target-1, `2`: target-2 |

```
/data 
┣ 📂 TaskA 
┃   ┣ 📂 EEG 
┃   ┃   ┣ 📜 S1_1.npz 
┃   ┃   ┣ 📜 S1_2.npz 
┃   ┃   ┣ 📜 ... 
┃   ┃   ┣ 📜 S43_5.npz 
┃   ┣ 📂 EM 
┃   ┃   ┣ 📜 S1_1.mat 
┃   ┃   ┣ 📜 S1_2.mat 
┃   ┃   ┣ 📜 ... 
┃   ┃   ┣ 📜 S43_5.mat

┣ 📂 TaskB 
┃   ┣ 📂 EEG 
┃   ┃   ┣ 📜 S1_1.npz 
┃   ┃   ┣ 📜 S1_2.npz 
┃   ┃   ┣ 📜 ... 
┃   ┃   ┣ 📜 S43_5.npz 
┃   ┣ 📂 EM 
┃   ┃   ┣ 📜 S1_1.mat 
┃   ┃   ┣ 📜 S1_2.mat 
┃   ┃   ┣ 📜 ... 
┃   ┃   ┣ 📜 S43_5.mat

┣ 📂 TaskC 
┃   ┣ 📂 EEG 
┃   ┃   ┣ 📜 S1_1.npz 
┃   ┃   ┣ 📜 S1_2.npz 
┃   ┃   ┣ 📜 ... 
┃   ┃   ┣ 📜 S43_5.npz 
┃   ┣ 📂 EM 
┃   ┃   ┣ 📜 S1_1.mat 
┃   ┃   ┣ 📜 S1_2.mat 
┃   ┃   ┣ 📜 ... 
┃   ┃   ┣ 📜 S43_5.mat
```

### 2.2&nbsp; Data Acquisition

The EEG data are recorded using a SynAmp2 Amplifier (NeuroScan, Australia) with a 64-channel Ag/AgCl electrode cap following the international 10-20 system. The electrode impedances are maintained below 10 kΩ, with AFz serving as the ground electrode and the vertex as the reference. Data are sampled at 1000 Hz. Both EEG and eye movement signals are recorded simultaneously during the experiment. 


### 2.3&nbsp; Data Preprocessing

In the preprocessing stage, the EEG data for each block are down-sampled to 250 Hz. Subsequently, the signals are filtered using a 3-order Butterworth filter with linear phase implementation between 0.1 and 15 Hz, eliminating slow drift and high-frequency noise while preventing delay distortions. Then, the continuous EEG data are segmented into trials from the onset of the presented image to 1000 ms after the onset, and the EEG data of -200-0 ms are used for baseline correction.  In the Task plane and people, each subject contains approximately 560 target samples and 13440 nontarget samples. In the Task car, each subject contains approximately 320 target samples and 7680 nontarget samples.


## 3&nbsp; Train

The MTREE-Net is optimized using the Adam optimizer. The model assessment follows within-subject decoding settings using the cross-validation strategy. Each subject consists of five blocks, with each block containing 1,000 EEG-EM sample pairs, where around 3% are target-1 samples, 3% are target-2 samples, and the remaining 94% are non-target samples. For each subject, we employ a 5-fold cross-validation to partition experimental blocks into training and test sets. Each block serves as the test set once, while the remaining blocks form the training set. Each fold contains approximately 30 target-1 samples, 30 target-2 samples, and 940 non-target samples. Within each training set,
a secondary 5-fold cross-validation is applied to subdivide the training data into training and validation sets.

```bash
python -m torch.distributed.launch --master_port 29502 --nproc_per_node=2 /EMLPformer/main.py
```

## 4&nbsp; Cite

If you find this code or our ELIPformer paper helpful for your research, please cite our paper:

```bibtex
@article{li2025integrating,
  title={Integrating Language-Image Prior into EEG Decoding for Cross-Task Zero-Calibration RSVP-BCI},
  author={Li, Xujin and Wei, Wei and Qiu, Shuang and Zhang, Xinyi and Li, Fu and He, Huiguang},
  journal={arXiv preprint arXiv:2501.02841},
  year={2025}
}
```
