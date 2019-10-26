### Train
* Weights：
<p>&emsp;&emsp;CRNNet weights are released.</p>

<p>&emsp;&emsp;CRDNet：https://drive.google.com/open?id=1H9CqWKZZwLn8V-nOWjCopyBRzKhZDTvo</p>

* CRDNet: 
```bash
python train.py --cuda --gpus=4 --train=/path/to/train --test=/path/to/test --lr=0.0001 --step=1000 --n 2
```

### Test
```bash
python test.py --cuda --checkpoints=/path/to/checkpoint --test=/path/to/testimages
```
