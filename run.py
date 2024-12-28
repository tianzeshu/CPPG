import os
import hydra
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from hydra import utils
from torch.utils.data import DataLoader
from models.model import PromptBartModel, PromptGeneratorModel
from models.teacher_model import TeacherPromptBartModel, TeacherPromptGeneratorModel
from module.datasets import ConllNERProcessor, ConllNERDataset
from module.train import Trainer
from module.metrics import Seq2SeqSpanMetric
from utils.util import get_loss, set_seed
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_CLASS = {
    'conll2003': ConllNERDataset,
}

DATA_PROCESS = {
    'conll2003': ConllNERProcessor,
}

DATA_PATH = {
    'conll2003': {'train': 'data/conll2003/answers.txt',
                  'dev': 'data/conll2003/dev.txt',
                  'test': 'data/conll2003/test.txt'},
}

MAPPING = {
    'conll2003': {'loc': '<<location>>',
                  'per': '<<person>>',
                  'org': '<<organization>>',
                  'misc': '<<others>>'},
}


@hydra.main(config_path="./conf", config_name="config.yaml")
def main(cfg):
    torch.set_default_dtype(torch.float16)
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    print(cfg)
    data_path = DATA_PATH[cfg.dataset_name]
    for mode, path in data_path.items():
        data_path[mode] = os.path.join(cfg.cwd, path)
    dataset_class, data_process = DATASET_CLASS[cfg.dataset_name], DATA_PROCESS[cfg.dataset_name]
    mapping = MAPPING[cfg.dataset_name]

    set_seed(cfg.seed)
    if cfg.save_path is not None:
        cfg.save_path = os.path.join(cfg.save_path, cfg.dataset_name + "_" + str(cfg.batch_size) + "_" + str(cfg.learning_rate) + cfg.notes)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path, exist_ok=True)
    if 'chinese' in cfg.bart_name:
        cfg.bart_name = os.path.join(utils.get_original_cwd(), cfg.bart_name)

    process = data_process(data_path=data_path, mapping=mapping, bart_name=cfg.bart_name, learn_weights=cfg.learn_weights)
    label_ids = list(process.mapping2id.values())
    teacher_model_list = []
    for worker_id in range(1, 48):
        teahcer_prompt_model = TeacherPromptBartModel(tokenizer=process.tokenizer, label_ids=label_ids, args=cfg)
        teacher_model = TeacherPromptGeneratorModel(prompt_model=teahcer_prompt_model, bos_token_id=0,
                                                    eos_token_id=1,
                                                    max_length=cfg.tgt_max_len, max_len_a=cfg.src_seq_ratio, num_beams=cfg.num_beams, do_sample=False,
                                                    repetition_penalty=1, length_penalty=cfg.length_penalty, pad_token_id=1,
                                                    restricter=None)
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        teacher_model = teacher_model.to(device)
        load_path = f"./save_model/annotator{worker_id}_best_model.pth"
        load_model_dict = torch.load(load_path)
        model_dict = teacher_model.state_dict()
        for name in load_model_dict:
            if name in model_dict:
                if model_dict[name].shape == load_model_dict[name].shape:
                    model_dict[name] = load_model_dict[name]
                else:
                    pass
            else:
                pass
        teacher_model.load_state_dict(model_dict)
        teacher_model_list.append(teacher_model)

    prompt_model = PromptBartModel(tokenizer=process.tokenizer, label_ids=label_ids, args=cfg)
    model = PromptGeneratorModel(prompt_model=prompt_model, bos_token_id=0,
                                 eos_token_id=1,
                                 max_length=cfg.tgt_max_len, max_len_a=cfg.src_seq_ratio, num_beams=cfg.num_beams, do_sample=False,
                                 repetition_penalty=1, length_penalty=cfg.length_penalty, pad_token_id=1,
                                 restricter=None)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataloader_list = []
    for worker_id in range(1, 48):
        train_dataset = dataset_class(data_processor=process, mode='train', load_worker_id=worker_id)
        train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=cfg.batch_size, num_workers=4)
        train_dataloader_list.append(train_dataloader)

    dev_dataset = dataset_class(data_processor=process, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, collate_fn=dev_dataset.collate_fn, batch_size=4, num_workers=4)

    metrics = Seq2SeqSpanMetric(eos_token_id=1, num_labels=len(label_ids), target_type='word')
    loss = get_loss
    trainer = Trainer(train_data=train_dataloader_list, dev_data=dev_dataloader, test_data=None, model=model, args=cfg, logger=logger, loss=loss,
                      metrics=metrics, device=device, teacher_model_list=teacher_model_list)
    trainer.train()


if __name__ == "__main__":
    main()
