# ConFusionformer
This is an implementation of "ConFusionformer: Locality-enhanced Conformer Through Multi-resolution Attention
Fusion for Speaker Verification".

## Requirements
Python 3.10, Pytorch 2.3.1, Pandas, Scipy, Timm

## Running examples
### Prepare training meta info and trails
```sh
bash prep.sh
```
This script will create a meta folder and a trials folder under the current directory from the corpus data, which is default to "../corpus/".
Please refer to utils/prep_vox_meta.py for the produced meta info files and utils/prep_vox_trails.py for the output trail files. These meta 
info and trials will be created only once and then used for embedding training. 

For your convenience, the produced meta and trials folders are available at https://drive.google.com/drive/folders/14OjUvTDaDoGG93OOX6HiaOqelO7y06WC?usp=sharing.

### Embedding training
```sh
bash train.sh
```
Running this code will create a "model_ckpt" directory, and the checkpoints will be saved in this folder.

### Embedding extraction
```sh
bash extract.sh
```
This code will extract embeddings using the saved checkpoint and save the embeddings under "eval/xvectors/".

### Performance evaluation
```sh
bash -m be_eval.sh
```
Cosine scores will be produced under "eval/scores/" and then EER and minDCF will be computed.

## References
```bibtex
@article{Tu25-confusionformer,
  title   = {ConFusionformer: Locality-enhanced Conformer Through Multi-resolution Attention
Fusion for Speaker Verification},
  author  = {Y. Tu and M. W. Mak and K. A. Lee and W. Lin},
  journal = {Neurocomputing},
  year    = {2025}
}
```
