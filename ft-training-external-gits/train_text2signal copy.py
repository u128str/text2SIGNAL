from typing import get_args

import hydra
import lightning as L
import urllib3
from hydra.core.config_store import ConfigStore
from lightning import LightningDataModule

from fine_tuning.config.training_config import InstructionTuningDataConfig
from fine_tuning.config.utils import validate_and_cast_config
from fine_tuning.data.instruction_tuning.dataloader_utils import prepare_data
from fine_tuning.training.lightning_wrappers import get_lightning_model

urllib3.disable_warnings()
from dataclasses import asdict

from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from fine_tuning.config.training_config import InstructionTuningConfig
from fine_tuning.data.data_module import DoNothingReader, LabDataModule, SplitKeys
from fine_tuning.training.utils import get_tokenizer, get_trainer

cs = ConfigStore.instance()
cs.store(name='text2signal_config', node=InstructionTuningConfig)


def get_data_module(tokenizer: AutoTokenizer, config: InstructionTuningDataConfig, debug=False) -> LightningDataModule:
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



@hydra.main(config_path='../fine_tuning/config', config_name='text2signal_config_signal')
@validate_and_cast_config(InstructionTuningConfig)
def main_local_instruction_tuning(cfg: InstructionTuningConfig):
    print(asdict(cfg))
    L.seed_everything(42)

    trainer = get_trainer(cfg)
    data_module = get_data_module(tokenizer=get_tokenizer(cfg.model.hf_model_id), config=cfg.data)
    ptl_model = get_lightning_model(cfg, trainer, data_module)

    trainer.fit(ptl_model, data_module)


if __name__ == '__main__':
    import sys
    sys.argv = [arg for arg in sys.argv if not arg.startswith('--local_rank')]
    hf_instruction_tuning = False
    # Decide which config you want to load:
    if hf_instruction_tuning == True:
        main_hf_instruction_tuning()
    else:
        main_local_instruction_tuning()
