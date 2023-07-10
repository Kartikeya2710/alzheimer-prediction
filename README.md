# <div align="center">Alzheimer Prediction</div>

Custom implementations of ResNeXt and EfficientNet providing training, logging and inference on [Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

## <div align="center">Documentation</div>

See below for a quickstart installation and usage example

<details open>
<summary>Install</summary>

1. Git clone the repository
```bash
git clone https://github.com/Kartikeya2710/alzheimer-prediction.git
```

2. Install the requirements from `requirements.txt`
```bash
pip3 install -r requirements.txt
```

</details>

<details open>
<summary>Usage</summary>

#### CLI

You can use it directly from the Command Line Interface (CLI):

```bash
python3 main.py --model-config configs/models/resnext.yaml --dataset-config configs/datasets/alzheimer.yaml
```

Note: You can set the `mode` of your model to `train` or `test` in its config file
<h5 a><strong><code>model.yaml</code></strong></h5>

```yaml
mode: train # or test
```

</details>

<details open>
<summary>Benchmark</summary>

| Model                                                                                | size<br><sup>(pixels) | params<br><sup>(M) | train_acc<br><sup>(%) | val_acc<br><sup>(%) | epochs
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ----------------- | ---------------- | ------------------------ |
| [ResNeXt-50](/graphs/models/ResNeXt/) | 224 x 224              | 23.015 | 99.01 | 98.90 | 20

</details>

<details open>
<summary>Logs</summary>

Example of logs for validation:
```bash
[INFO] - 2023-06-21 22:11:26,009 - root - : Using resnext... in ./alzheimer-prediction/agents/alzheimer.py:23
[INFO] - 2023-06-21 22:11:26,090 - root - : Created annotations at ./data/imageset.csv in ./alzheimer-prediction/agents/alzheimer.py:26
[INFO] - 2023-06-21 22:11:26,514 - root - : Operation will be on *****GPU-CUDA*****  in ./alzheimer-prediction/agents/alzheimer.py:53
[INFO] - 2023-06-21 22:11:29,144 - root - : Loading checkpoint 'experiments/resnextalzheimer/checkpoints/model_best.pth.tar' in ./alzheimer-prediction/agents/alzheimer.py:93
[INFO] - 2023-06-21 22:11:29,663 - root - : Checkpoint loaded successfully from 'experiments/resnextalzheimer/checkpoints/' at (epoch 21) in ./alzheimer-prediction/agents/alzheimer.py:100
[INFO] - 2023-06-21 22:11:38,909 - root - : Validation result at epoch-21 | - Val Acc: tensor(98.9114, device='cuda:0') in ./alzheimer-prediction/agents/alzheimer.py:219
```

</details>