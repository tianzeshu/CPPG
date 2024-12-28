import torch
import logging
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class ConllNERProcessor(object):
    def __init__(self, data_path, mapping, bart_name, learn_weights) -> None:
        self.data_path = data_path
        self.bart_name = bart_name
        self.tokenizer = BertTokenizer.from_pretrained(self.bart_name)
        self.mapping = mapping
        self.original_token_nums = self.tokenizer.vocab_size
        self.learn_weights = learn_weights
        self._add_tags_to_tokens()

    def load_from_file(self, mode='train', load_worker_id=0):
        """load conll ner from file

        Args:
            mode (str, optional): train/test/dev. Defaults to 'train'.
        Return:
            outputs (dict)
            raw_words: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
            raw_targets: ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
            entities: [['EU'], ['German'], ['British']]
            entity_tags: ['org', 'misc', 'misc']
            entity_spans: [[0, 1], [2, 3], [6, 7]]
        """
        crowd = True
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))

        split_c = '\t' if 'conll' in load_file else ' '
        outputs = {'raw_words': [], 'raw_targets': [], 'entities': [], 'entity_tags': [], 'entity_spans': [], 'workers': []}

        if not crowd or mode != "train":
            with open(load_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                raw_words, raw_targets = [], []
                raw_word, raw_target = [], []
                raw_workers = []
                for line in lines:
                    if line != "\n":
                        raw_word.append(line.split(split_c)[0])
                        raw_target.append(line.split(split_c)[1][:-1])
                    else:
                        raw_words.append(raw_word)
                        raw_targets.append(raw_target)
                        raw_workers.append([0])
                        raw_word, raw_target = [], []
        else:
            if load_worker_id == 0:
                sentence = []

                sentences = []
                all_labels = []
                labels = []
                worker = []
                workers = []
                with open(load_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line == "\n":
                            sentences.append(sentence)
                            all_labels.append(labels)
                            workers.append(worker)
                            labels = []
                            sentence = []
                            worker = []
                        else:
                            temp = line.strip().split()
                            word = temp[0]
                            temp_labels = temp[1:]

                            if len(labels) == 0:
                                label_dict = Counter(temp_labels)
                                if "?" in label_dict:
                                    label_dict.pop("?")
                                sum = 0
                                for k, v in label_dict.items():
                                    sum += v
                                for i in range(sum):
                                    labels.append([])
                            sentence.append(word)
                            index = 0
                            for i, label in enumerate(temp_labels):
                                if label != "?":
                                    labels[index].append(label)
                                    if i + 1 not in worker:
                                        worker.append(i + 1)
                                    index += 1

                assert len(sentences) == len(all_labels) == len(workers)
                raw_words, raw_targets, raw_workers = [], [], []
                for i in range(len(sentences)):
                    for j in range(len(all_labels[i])):
                        raw_words.append(sentences[i])
                        raw_targets.append(all_labels[i][j])
                        raw_workers.append(workers[i][j])
            else:
                sentence = []

                sentences = []
                all_labels = []
                labels = []
                worker = []
                workers = []
                with open(load_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line == "\n":
                            sentences.append(sentence)
                            all_labels.append(labels)
                            workers.append(worker)
                            labels = []
                            sentence = []
                            worker = []
                        else:
                            temp = line.strip().split()
                            word = temp[0]
                            temp_labels = temp[1:]

                            sentence.append(word)
                            for i, label in enumerate(temp_labels):
                                if label != "?" and i + 1 == load_worker_id:
                                    labels.append(label)
                                    # worker id 从1至47
                                    if i + 1 not in worker:
                                        worker.append(i + 1)

                assert len(sentences) == len(all_labels) == len(workers)
                raw_words, raw_targets, raw_workers = [], [], []
                for i in range(len(sentences)):
                    if len(workers[i]) != 0:
                        raw_words.append(sentences[i])
                        raw_targets.append(all_labels[i])
                        raw_workers.append(workers[i][0])

        for words, targets, worker in zip(raw_words, raw_targets, raw_workers):
            entities, entity_tags, entity_spans = [], [], []
            start, end, start_flag = 0, 0, False
            for idx, tag in enumerate(targets):
                if tag.startswith('B-'):
                    end = idx
                    if start_flag:
                        entities.append(words[start:end])
                        entity_tags.append(targets[start][2:].lower())
                        entity_spans.append([start, end])
                        start_flag = False
                    start = idx
                    start_flag = True
                elif tag.startswith('I-'):
                    end = idx
                elif tag.startswith('O'):
                    end = idx
                    if start_flag:
                        entities.append(words[start:end])
                        entity_tags.append(targets[start][2:].lower())
                        entity_spans.append([start, end])
                        start_flag = False
            if start_flag:  # 句子以实体I-结束，未被添加
                entities.append(words[start:end + 1])
                entity_tags.append(targets[start][2:].lower())
                entity_spans.append([start, end + 1])
                start_flag = False

            if len(entities) != 0:
                outputs['raw_words'].append(words)
                outputs['raw_targets'].append(targets)
                outputs['entities'].append(entities)
                outputs['entity_tags'].append(entity_tags)
                outputs['entity_spans'].append(entity_spans)
                outputs['workers'].append(worker)
        return outputs

    def process(self, data_dict):
        target_shift = len(self.mapping) + 2

        def prepare_target(item):
            raw_word = item['raw_word']
            word_bpes = [[self.tokenizer.cls_token_id if 'chinese' in self.bart_name else self.tokenizer.bos_token_id]]
            first = []
            cur_bpe_len = 1
            for word in raw_word:
                bpes = self.tokenizer.tokenize(word) if 'chinese' in self.bart_name else self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                first.append(cur_bpe_len)
                cur_bpe_len += len(bpes)
                word_bpes.append(bpes)
            assert first[-1] + len(bpes) == sum(map(len, word_bpes))
            word_bpes.append([self.tokenizer.sep_token_id if 'chinese' in self.bart_name else self.tokenizer.eos_token_id])
            assert len(first) == len(raw_word) == len(word_bpes) - 2

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(lens).tolist()

            entity_spans = item['entity_span']  # [(s1, e1, s2, e2), ()]
            entity_tags = item['entity_tag']  # [tag1, tag2...]
            entities = item['entity']  # [[ent1, ent2,], [ent1, ent2]]
            target = [0]
            pairs = []

            first = list(range(cum_lens[-1]))

            assert len(entity_spans) == len(entity_tags)  #
            for idx, (entity, tag) in enumerate(zip(entity_spans, entity_tags)):
                cur_pair = []
                num_ent = len(entity) // 2
                for i in range(num_ent):
                    start = entity[2 * i]
                    end = entity[2 * i + 1]
                    cur_pair_ = []
                    cur_pair_.extend([cum_lens[k] for k in list(range(start, end))])
                    cur_pair.extend([p + target_shift for p in cur_pair_])
                for _, (j, word_idx) in enumerate(zip((cur_pair[0], cur_pair[-1]), (0, -1))):
                    j = j - target_shift
                assert all([cur_pair[i] < cum_lens[-1] + target_shift for i in range(len(cur_pair))])

                cur_pair.append(self.mapping2targetid[tag] + 2)
                pairs.append([p for p in cur_pair])
            target.extend(list(chain(*pairs)))
            target.append(1)

            word_bpes = list(chain(*word_bpes))
            assert len(word_bpes) < 500

            dict = {'tgt_tokens': target, 'target_span': pairs, 'src_tokens': word_bpes,
                    'first': first, 'src_seq_len': len(word_bpes), 'tgt_seq_len': len(target), 'worker': item['worker']}
            return dict

        logger.info("Process data...")
        for raw_word, raw_target, entity, entity_tag, entity_span, raw_worker in tqdm(zip(data_dict['raw_words'], data_dict['raw_targets'], data_dict['entities'],
                                                                                          data_dict['entity_tags'], data_dict['entity_spans'], data_dict['workers']), total=len(data_dict['raw_words']),
                                                                                      desc='Processing'):
            item_dict = prepare_target({'raw_word': raw_word, 'raw_target': raw_target, 'entity': entity, 'entity_tag': entity_tag, 'entity_span': entity_span, "worker": raw_worker})
            # add item_dict to data_dict
            for key, value in item_dict.items():
                if key in data_dict:
                    data_dict[key].append(value)
                else:
                    data_dict[key] = [value]
        return data_dict

    def _add_tags_to_tokens(self):
        mapping = self.mapping
        if self.learn_weights:  # add extra tokens to huggingface tokenizer
            self.mapping2id = {}
            self.mapping2targetid = {}
            for key, value in self.mapping.items():
                key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value[2:-2]) if 'chinese' in self.bart_name else self.tokenizer.tokenize(value[2:-2], add_prefix_space=True))
                self.mapping2id[value] = key_id  # may be list
                self.mapping2targetid[key] = len(self.mapping2targetid)
        else:
            tokens_to_add = sorted(list(mapping.values()), key=lambda x: len(x), reverse=True)  # 
            unique_no_split_tokens = self.tokenizer.unique_no_split_tokens  # no split
            sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x: len(x), reverse=True)
            for tok in sorted_add_tokens:
                assert self.tokenizer.convert_tokens_to_ids([tok])[0] == self.tokenizer.unk_token_id  #
            self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens  # add to no_split_tokens
            self.tokenizer.add_tokens(sorted_add_tokens)
            self.mapping2id = {}  # tag to id
            self.mapping2targetid = {}  # tag to number

            for key, value in self.mapping.items():
                key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
                assert len(key_id) == 1, value
                assert key_id[0] >= self.original_token_nums
                self.mapping2id[value] = key_id[0]  #
                self.mapping2targetid[key] = len(self.mapping2targetid)


class ConllNERDataset(Dataset):
    def __init__(self, data_processor, mode='train', load_worker_id=0) -> None:
        self.data_processor = data_processor
        self.data_dict = data_processor.load_from_file(mode=mode, load_worker_id=load_worker_id)
        self.complet_data = data_processor.process(self.data_dict)
        self.mode = mode

    def __len__(self):
        return len(self.complet_data['src_tokens'])

    def __getitem__(self, index):
        if self.mode == 'test':
            return torch.tensor(self.complet_data['src_tokens'][index]), torch.tensor(self.complet_data['src_seq_len'][index]), \
                torch.tensor(self.complet_data['first'][index]), self.complet_data['raw_words'][index]

        return torch.tensor(self.complet_data['src_tokens'][index]), torch.tensor(self.complet_data['tgt_tokens'][index]), \
            torch.tensor(self.complet_data['src_seq_len'][index]), torch.tensor(self.complet_data['tgt_seq_len'][index]), \
            torch.tensor(self.complet_data['first'][index]), self.complet_data['target_span'][index], torch.tensor(self.complet_data['worker'][index])

    def collate_fn(self, batch):
        src_tokens, src_seq_len, first = [], [], []
        tgt_tokens, tgt_seq_len, target_span = [], [], []
        worker = []
        if self.mode == "test":
            raw_words = []
            for tup in batch:
                src_tokens.append(tup[0])
                src_seq_len.append(tup[1])
                first.append(tup[2])
                raw_words.append(tup[3])
            src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=self.data_processor.tokenizer.pad_token_id)
            first = pad_sequence(first, batch_first=True, padding_value=0)
            return src_tokens, torch.stack(src_seq_len, 0), first, raw_words

        for tup in batch:
            src_tokens.append(tup[0])
            tgt_tokens.append(tup[1])
            src_seq_len.append(tup[2])
            tgt_seq_len.append(tup[3])
            first.append(tup[4])
            target_span.append(tup[5])
            worker.append(tup[6])
        src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=self.data_processor.tokenizer.pad_token_id)
        tgt_tokens = pad_sequence(tgt_tokens, batch_first=True, padding_value=1)
        first = pad_sequence(first, batch_first=True, padding_value=0)
        return src_tokens, tgt_tokens, torch.stack(src_seq_len, 0), torch.stack(tgt_seq_len, 0), first, target_span, torch.stack(worker, 0)
