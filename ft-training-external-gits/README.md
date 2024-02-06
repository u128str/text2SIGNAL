# Fine-Tuning set-up


Instructions on how to FT mistral, codellama and other Hagginsface opensource models:

1. Clone SAP repo with FT solution [llm-fine-tuning-lab](https://github.tools.sap/DL-COE/llm-fine-tuning-lab) (in needed): 
   ```bash
   $ mkdir ft-training-external-gits
   $ cd ft-training-external-gits
   ft-training-external-gits$ git submodule add https://github.tools.sap/DL-COE/llm-fine-tuning-lab
   ```

Next step is to modify copy/paste relevant files to proper locations:
```js
ft-training-external-gits/
├── llm-fine-tuning-lab/ 
├── README.md   
├── train_text2signal.py           -> llm-fine-tuning-lab/scripts/train_text2signal.py
├── generate_predictions.py        -> llm-fine-tuning-lab/fine_tuning/evaluation/generate_predictions.py
├── lora_starcoder.yaml            -> llm-fine-tuning-lab/fine_tuning/config/trainer/lora_starcoder.yaml (depricated)
├── text2signal_config_alexey.yaml -> llm-fine-tuning-lab/fine_tuning/config/text2signal_config_alexey.yaml
└── text2signal_tuning_data.yaml   -> llm-fine-tuning-lab/fine_tuning/config/data/text2signal_tuning_data.yaml
```

### Training - FT Hugging Face LLMs: mistralai/Mistral-7B-Instruct-v0.1 codellama/CodeLlama-7b-Instruct-hf

We assume that respective docker image is on DGX cloud and this dir is mounted to the DGX systems external workspace.

Run DGX job
```bash
ft-coe$ ngc workspace upload --source text2signal  text2signal
ft-coe$ ngc workspace upload --source llm-fine-tuning-lab  aa-all

ngc base-command job run --name "Job-sap-scus-ace-622938" --priority NORMAL --preempt RUNONCE --min-timeslice 7776000s --total-runtime 7776000s --ace sap-scus-ace --instance dgxa100.80g.1.norm --commandline "set -x && jupyter lab --allow-root --ip=* --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.allow_origin='*' --notebook-dir=/ --ContentsManager.allow_hidden=True & sleep infinity" --result /results --image "l9lkbxzngzmv/deepspeed:mpi4py_patched" --org l9lkbxzngzmv --workspace r0bfCtECTnGTPdOJqZ8dlQ:/mount/data:RW --workspace 1muSRyCuQTyyx0pB1-jXuw:/mount/ftcoe:RW --port 8888
```


**llm-fine-tuning-lab/scripts/train_text2signal.py** 
The key file to run FT experiment (assuming that this folder is cloned on GPU node):

#### I. Step Fine-Tuning run 11 Epochs 20h

```bash
PYTHONPATH=. python scripts/train_text2signal.py model.hf_model_id=mistralai/Mistral-7B-Instruct-v0.1
# OR
PYTHONPATH=. python scripts/train_text2signal.py model.hf_model_id=codellama/CodeLlama-7b-Instruct-hf
```

#### II. Step merge adapters with original weights to one repo:

```bash
# 11 epoch 1536
PYTHONPATH=. python scripts/merge_peft_adapters.py --base_model_name_or_path codellama/CodeLlama-7b-Instruct-hf --peft_model_path /results/local_dataset_out_11_epochs_codellama-lr-05/codellama/CodeLlama-7b-Instruct-hf-lora-11-epochs/ckpts/epoch\=10-step\=15540-hf-ckpt/
```

#### III. Run inference to generate outputs with FT predictions:

```bash
PYTHONPATH=. python fine_tuning/evaluation/generate_predictions.py  --model_id codellama/CodeLlama-7b-Instruct-hf --model_path /results/local_dataset_out_30_epochs_codellama-lr-05/codellama/CodeLlama-7b-Instruct-hf-lora-30-epochs/ckpts/epoch=29-step=42191-hf-ckpt-merged --gt_path /mount/data/text2signal_test_1000.jsonl --output_path out_code_llame-epochs-30-lr-05.json --llm_name codellama
```


## Fine-Tuning with Aleph-alpha

Assumptions that docker file and respective image are available:

```bash
docker pull registry.gitlab.aleph-alpha.de/product/finetuning-deployment/aa-finetuning
docker image inspect registry.gitlab.aleph-alpha.de/product/finetuning-deployment/aa-finetuning:latest
docker image inspect bc0ca77efc7c
docker tag bc0ca77efc7c6a24e1cd04181a3c1e998dfb07820d5ad7b2e3d96c7c4e003df4 nvcr.io/l9lkbxzngzmv/aa-ft-01:latest

```

However for DGX set up one need add jupyter lab to the above docker *DokerfileAA.dgxcloud* image:

 ```docker
FROM nvcr.io/l9lkbxzngzmv/aa-ft-01

RUN pip install jupyterlab
```

Next we build it 

```bash
docker build -f DokerfileAA.dgxcloud .
```

re-tag and push to DGX system:

```bash
docker tag 797ea06b07c1 nvcr.io/l9lkbxzngzmv/aa-ft-02:latest
docker  push nvcr.io/l9lkbxzngzmv/aa-ft-02:latest 
```

Run DGX JOB:

```bash
ngc base-command job run --name "Job-sap-scus-ace-637221" --priority NORMAL --preempt RUNONCE --min-timeslice 7776000s --total-runtime 7776000s --ace sap-scus-ace --instance dgxa100.80g.1.norm --commandline "set -x && jupyter lab --allow-root --ip=* --port=8889 --no-browser --NotebookApp.token='' --NotebookApp.allow_origin='*' --notebook-dir=/ --ContentsManager.allow_hidden=True & sleep infinity" --result /results --image "l9lkbxzngzmv/aa-ft-02:latest" --org l9lkbxzngzmv --workspace r0bfCtECTnGTPdOJqZ8dlQ:/mount/ftaa:RW --port 8889
```

Login to the Jupyter lab from above job and run fine-tuning with AA:

```bash
/aleph_alpha_luminous# python run.py /aleph_alpha_luminous/configs/luminous_base_adapter.yml 
```


