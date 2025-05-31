# Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation

Code implementation for [Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation](https://openreview.net/forum?id=Wz2T778EKJ) (*ICML 2025*).

## 👨‍💻 Quick Usage
```
python main.py --dataset ImageNet --datadir /home/path-to-datasets/ --savedir /home/path-to-savedir/ --net_arch resnet34_qcfs --amp --batchsize 100 --dev 0 --time_step 8 --neuron_type ParaInfNeuron --checkpoint_path /home/path-to-checkpoints --pretrained_model --direct_inference
```

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝:

```bibtex
@inproceedings{hao2025conversion,
  title={Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation},
  author={Hao, Zecheng and Ma, Qichao and Chen, Kang and Zhang, Yi and Yu, Zhaofei and Huang, Tiejun},
  year={2025},
  booktitle={International Conference on Machine Learning}
}
```

