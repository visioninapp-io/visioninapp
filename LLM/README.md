# install
## python
python version = 3.11.9

## CUDA, torch
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

## requirements
pip install -r requirements.txt

# Setup
## Gpt-5 api
always temperature=1

## .env
OPEN-API-KEY = {Your-api key}
OPEN-API-BASE = {api endpoint}

## Architecture
```
├─.venv
├─configs
│  ├─labeling.yaml
│  └─training.yaml
├─data
│  ├─datasets
│  ├─models
│  └─runs
├─graph
│  ├─labeling
│  │  └─node
│  └─training
│     └─nodes
├─llm
│  └─prompts
├─registry
├─tools
├─utils
├─.env
├─.gitignore
├─README.md
├─requirements.txt
└─run.py
```