# 🚀 Interaction-Guided Mixture of Experts for  Weakly Supervised Imbalanced Audio-Visual Video Parsing
This is the official code for the Interaction-Guided Mixture of Experts for  Weakly Supervised Imbalanced Audio-Visual Video Parsing.

![image](https://github.com/MMVAT/M2MOE/blob/main/arch.png?raw=true)


# 💻 Machine environment
- Ubuntu version: 20.04.6 LTS (Focal Fossa)
- CUDA version: 12.2
- PyTorch: 1.12.1
- Python: 3.10.12
- GPU: NVIDIA A100-SXM4-40GB
- [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
  [![CUDA](https://img.shields.io/badge/CUDA-11.2+-green.svg)](https://developer.nvidia.com/cuda-zone)

# 🛠 Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/MMVAT/TAMoE.git
cd TAMoE
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate tamoe
```

# 📂 Data Preparation
### Annotation files
Please download LLP dataset annotations (6 CSV files) from [AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20) and place them in ```data/r2plus1d_18/.```

### CLAP- & CLIP-extracted features
Please download the CLAP-extracted features (CLAP.7z) and CLIP-extracted features (CLIP.7z) from [this link](https://pan.quark.cn/s/db27c79f651b?pwd=rF5C), unzip the two files, and place the decompressed CLAP-related files in ```data/CLAP/.``` and the CLIP-related files in ```data/CLIP/.```

### File structure for datasets
Please make sure that the file structure is the same as the following.
```bash
data/                                
│   ├── AVVP_dataset_full.csv               
│   ├── AVVP_eval_audio.csv             
│   ├── AVVP_eval_visual.csv                 
│   ├── AVVP_test_pd.csv                
│   ├── AVVP_train.csv                     
│   ├── AVVP_val_pd.csv                      
│   ├── CLIP/                                
│   │   ├── features/        
│   │   │   ├── -00BDwKBD5i8.npy
│   │   │   ├── -00fs8Gpipss.npy
│   │   │   └── ... 
│   │   ├── segment_pseudo_labels/        
│   │   │   ├── -00BDwKBD5i8.npy
│   │   │   ├── -00fs8Gpipss.npy
│   │   │   └── ... 
│   ├── CLAP/              
│   │   ├── features/        
│   │   │   ├── -00BDwKBD5i8.npy
│   │   │   ├── -00fs8Gpipss.npy
│   │   │   └── ... 
│   │   ├── segment_pseudo_labels/        
│   │   │   ├── -00BDwKBD5i8.npy
│   │   │   ├── -00fs8Gpipss.npy
│   │   │   └── ... 
│   ├── r2plus1d_18/              
│   │   ├── -00BDwKBD5i8.npy
│   │   ├── -00fs8Gpipss.npy
│   │   └── ... 
```

# 🎓 Download trained models
Please download the trained models from [this link](https://pan.quark.cn/s/f9f220c0e73d?pwd=mRZ2) and put the models in their corresponding model directory.

# 🔥 Training and Inference
We provide bash file for a quick start.
#### For Training
```
bash train.sh
```

#### For Inference
```
bash test.sh
```

# 🤝 Contributing
We welcome contributions to the M2MOE project! To contribute:

1. **Fork the repository** and create a new branch for your feature
2. **Ensure your code follows the project's coding standards**
3. **Add tests** for any new functionality
4. **Submit a pull request** with a clear description of your changes

# 📞 Contact & Acknowledgements

## Contact Information

- **Primary Contact**: [isahini@csuft.edu.cn]
- **GitHub Issues**: [Submit issues](https://github.com/MMVAT/M2MOE/issues)
- **Discussions**: [Join discussions](https://github.com/MMVAT/M2MOE/discussions)

## Acknowledgements

We extend our gratitude to the following organizations and research groups:

- **LAION Team** for providing the CLAP audio-visual pre-trained models
- **AVVP Dataset Contributors** for the comprehensive audio-visual event dataset
- **PyTorch Team** for the robust deep learning framework
- **Open Source Community** for the invaluable contributions to multi-modal learning research

---

<div align="center">

**If this project contributes to your research, please consider giving it a ⭐ Star!**

</div>
