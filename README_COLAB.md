# Guide for Training on Google Colab with GPU

## Step 1: Open Notebook on Colab

1. Open Google Colab: https://colab.research.google.com/
2. Upload the file `notebooks/01_Feature_Learning.ipynb` to Colab
   - File → Upload notebook → Select file

## Step 2: Enable GPU

1. In Colab, click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator** → **GPU** (T4 or V100)
3. Click **Save**

## Step 3: Run Cells in Notebook

The notebook is designed to:
- Automatically detect and use GPU
- Guide you through uploading required files
- Run training with GPU acceleration

## Files to Upload

1. **src/models/autoencoder.py** - Model definition
2. **src/train_ae.py** - Training script
3. **data/processed/radioml_2018_processed.npz** - Processed data

## Important Notes

### About GPU:
- Code automatically detects GPU: `'device': 'cuda' if torch.cuda.is_available() else 'cpu'`
- If GPU is enabled, code will automatically use GPU
- You can check GPU status using the first cell in the notebook

### About Colab Limits:
- **Free tier**: ~12 hour sessions, may disconnect
- **Solutions**: 
  - Download model immediately after training completes
  - Or save to Google Drive (requires mounting drive)
  - Or use Colab Pro for longer sessions

### Performance Optimization:
- Increase `batch_size` if GPU has more memory (e.g., 512 instead of 256)
- Monitor GPU usage: `!nvidia-smi`
- Reduce number of epochs if you just want to test quickly

## Troubleshooting

### GPU Not Detected:
- Check if Runtime type has GPU selected
- Restart runtime: Runtime → Restart runtime
- Re-run the GPU check cell

### Out of Memory:
- Reduce `batch_size` in CONFIG
- Reduce number of epochs
- Use gradient checkpointing (if needed)

### Files Not Found:
- Check if file paths are correct
- Ensure all required files have been uploaded
- Verify directory structure

## After Training Completes

1. Download model: `saved_models/ae_weights.pth`
2. Save to local machine or Google Drive
3. Use this model for Phase 2 (train classifier)
