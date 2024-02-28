from typing import Optional, get_args

import hydra
import lightning as L
import urllib3
from hydra.core.config_store import ConfigStore
from lightning import LightningDataModule
from omegaconf import OmegaConf
from pydantic.dataclasses import dataclass

from fine_tuning.config.utils import validate_and_cast_config
from fine_tuning.data.dict.dataloader_utils import DictDataConfig, prepare_data
from fine_tuning.training.trainer import get_lightning_model

urllib3.disable_warnings()
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from fine_tuning.config.training_config import Config
from fine_tuning.data.data_module import DoNothingReader, LabDataModule, SplitKeys
from fine_tuning.training.utils import get_trainer


@dataclass
class TrainConfigDict(Config):
    data: DictDataConfig


cs = ConfigStore.instance()
cs.store(name='text2signal_config', node=TrainConfigDict)


def get_data_module(tokenizer: AutoTokenizer, config: DictDataConfig, debug=False) -> LightningDataModule:
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    data, to_output = prepare_data(config, tokenizer=tokenizer)
    data_module = LabDataModule(collate_fn=data_collator, num_workers=config.num_workers, debug=debug)
    for split in get_args(SplitKeys):
        data_module.append_data(split,
                                dataset_list=[data[split]],
                                transforms=[],
                                to_output=to_output,
                                num_workers=config.num_workers,
                                batch_size=data_module.batch_size,
                                reader=DoNothingReader())

    return data_module

@hydra.main(config_path='../fine_tuning/config', config_name='text2signal_config_alexey')
@validate_and_cast_config(TrainConfigDict)
def main_local_jsonl(cfg: TrainConfigDict):
    print(OmegaConf.to_container(cfg))
    L.seed_everything(42)

    ptl_model = get_lightning_model(cfg)
    trainer = get_trainer(cfg, custom_callbacks=[])
    data_module = get_data_module(tokenizer=ptl_model.tokenizer, config=cfg.data)

    trainer.fit(ptl_model, data_module)


if __name__ == '__main__':
    import sys
    sys.argv = [arg for arg in sys.argv if not arg.startswith('--local_rank')]
    hf_dataset = False
    # Decide which config you want to load:
    if hf_dataset == True:
        main_hf_dataset()
    else:
        main_local_jsonl()
