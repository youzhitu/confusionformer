# confusionformer
This is an implementation of "ConFusionformer: Locality-enhanced Conformer Through Multi-resolution Attention
Fusion for Speaker Verification".

## Requirements
Python 3.10, Pytorch 2.30, Pandas, Scipy, Timm

## Running examples
### Embdding training
```sh
python train.py
```
This code will load VoxCeleb2-dev, MUSAN, and RIR from "../corpus", the default directory of corpora. To simplify data loading,
meta information files containing path, utt_id, spk_id, No. of samples, duration, and speaker label are prepared and saved under
the "meta" directory. Running this code will create a "model_ckpt" directory and the checkpoints will be saved in this folder.

### Embdding extraction
```sh
python extract.py --ckpt_dir=model_ckpt/ckpt_20240503_0142 --ckpt_num=1
```
This code will extract embeddings using the saved checkpoint and save the embeddings under "eval/xvectors".

### Performance evaluation
```sh
python -m be_run.be_voxceleb
```
Based on the extracted embeddings, Cosine scores will be produced under "eval/scores" and EER and minDCF will be computed.

## References
```bibtex
@article{Jin25-adv_temp_dkd,
  title   = {ConFusionformer: Locality-enhanced Conformer Through Multi-resolution Attention
Fusion for Speaker Verification},
  author  = {Y. Tu and M. W. Mak and K. A. Lee and W. Lin},
  journal = {Neurocomputing},
  year    = {2025}
}
```
