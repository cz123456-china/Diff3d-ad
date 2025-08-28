# Diff3D-AD

## Usage
### installation
pip install requirements.txt

### Checkpoints
Due to the large file size, the weights are not uploaded here and will be provided later. The testing and training code is provided here. A more complete version of the code will be released later."

## Tran 3D
For training you can use the train_3d.py 
```
python train_3d.py --gpu_id 0 --obj_id $i --lr 0.0002 --bs 16 --epochs 120 --data_path $MVTEC3D_PATH --out_path $OUT_PATH --run_name $RUN_NAME
```

### Evaluate
```
python test_3d.py --gpu_id 0 --data_path $MVTEC3D_PATH --out_path $OUT_PATH --run_name $RUN_NAME 
```

## Train 2D
Please specify the dataset path(MVTec-3D-RGB), anomaly_source_path(DTD), and output folder in args.json and run:
```bash
python train_2D.py
```
## Evaluation
To perform inference with checkpoints, please run:
```bash
python eval.
```
## Test 3D-2D Fusion
To evaluate the final performance of the fused 3D and 2D model, please run:
python test.py 

**Parameters Description:**
- \`--checkpoint_3d\`: Path to the trained 3D model checkpoint (.pckl or .pth)
- \`--checkpoint_2d\`: Path to the trained 2D model checkpoint
