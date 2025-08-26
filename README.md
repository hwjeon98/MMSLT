# MMSLT

## Installation

Follow the steps below to set up the environment and dependencies for the project:

1.  **Create a Conda Environment**

Create a new Conda environment named `mmslt` with Python version 3.8.0:

`conda create -n mmslt python=3.8.0`

2.  **Activate the Conda Environment**

Activate the newly created environment:

`conda activate mmslt`

3.  **Install Dependencies**

Install the required dependencies using `requirements.txt`:

`pip install -r requirements.txt`

## Code Descriptions

### 1. **MMLP Training**

  
To train the MMLP (MultiModal Language Processing) model, run the following command:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    
    --nproc_per_node=4 \
    --master_port=1234 \
    --use_env train_mmlp.py \
    --batch-size 4 \
    --epochs 80 \
    --opt adamw \ 
    --lr 1e-4 \ 
    --output_dir /path/to/your/path/pretrain_models/mmlp

- Replace `/path/to/your/path/pretrain_models/mmlp` with the directory path where you want to save the MMLP model checkpoints and outputs.

----------

### 2. **MMSLT Training**

To fine-tune the MMSLT (MultiModal Spoken Language Translation) model, run the following command:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=1234 \
    --use_env train_mmslt.py \
    --batch-size 2 \
    --epochs 200 \
    --opt adamw \
    --lr 1e-4 \
    --finetune /path/to/your/path/pretrain_models/mmlp/best_checkpoint.pth \
    --output_dir /path/to/your/path/out/mmslt

- Replace `/path/to/your/path/pretrain_models/mmlp/best_checkpoint.pth` with the path to the best checkpoint from the MMLP training process.

- Replace `/path/to/your/path/out/mmslt` with the directory path where you want to save the MMSLT model checkpoints and outputs.

## Notes
  
- The `--nproc_per_node=4` flag specifies that the training will use 4 GPUs. Adjust this based on your available GPU resources. However, for exact reproducibility of the results, it is highly recommended to use 4 GPUs as specified.

----------
>>>>>>> d43ce32 (add submission code)
