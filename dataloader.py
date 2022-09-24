from util import *
import torch.utils.data as util_data
from torch.utils.data import Dataset
from data_augumentation import *
import jsonlines


def set_seed(seed):
    random.seed(seed)
    np.random.seed(10)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
class Data:
    
    def __init__(self, args):
        # 随机初始化
        set_seed(args.seed)

        # 载入数据集的关键信息
        max_seq_lengths = {'clinc':30, 'banking':55, 'snips': 35, "HWU64":25, "virtual_assistant":55}
        args.max_seq_length = max_seq_lengths[args.dataset]

        # 随机选取已知类和未知类(得到IND和OOD的类别list)
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list_pretraining = self.get_labels(self.data_dir)
        print("the numbers of all known labels:", len(self.all_label_list_pretraining))
        print(self.all_label_list_pretraining)

        self.data_dir_test = os.path.join(args.data_dir, args.dataset_test)
        self.all_label_list_clustering = self.get_labels(self.data_dir_test)
        print("the numbers of all unknown labels:", len(self.all_label_list_clustering))
        print(self.all_label_list_clustering)

        self.n_known_cls = len(self.all_label_list_pretraining)
        self.n_unknown_cls = len(self.all_label_list_clustering)

        # 载入数据集(tsv文件的表格，二维列表形式)
        pretraining_train_sets = self.get_datasets(self.data_dir, 'train')
        pretraining_eval_sets = self.get_datasets(self.data_dir, 'eval')

        clustering_train_sets = self.get_datasets(self.data_dir_test, 'train')

        test_set = self.get_testsets("data/test-utterances.jsonl", 'test')

        # (此时仍然还是字符形式的)
        self.pretrain_all_examples = self.get_samples(pretraining_train_sets, args, "train")
        self.pretrain_all_examples_eval = self.get_samples(pretraining_eval_sets, args, "train")
        self.cluster_all_examples = self.get_samples(clustering_train_sets, args, "train")
        self.test_all_examples = self.get_samples(test_set, args, "test")


        # 封装成dataloader格式(此时需要vectorization)
        self.pretrain_dataloader = self.get_loader_v2(self.pretrain_all_examples,self.all_label_list_pretraining,args,"train")
        self.pretrain_dataloader_eval = self.get_loader_v2(self.pretrain_all_examples_eval,self.all_label_list_pretraining, args, "eval")
        self.clustering_dataloader = self.get_loader_v2(self.cluster_all_examples, self.all_label_list_clustering, args,"train")
        self.test_dataloader = self.get_loader_v2(self.test_all_examples, self.all_label_list_clustering, args,
                                                    "test")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, mode = 'train', quotechar=None):
        with open(os.path.join(data_dir, mode+".tsv"), "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i=0
            for line in reader:
                if (i==0):
                    i+=1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines

    def get_testsets(self, data_dir, mode = 'train'):
        lines = []
        #utterances, intent_labels = [], []
        with open(data_dir, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                lines.append([item['utterance'], item['intent']])
                #utterances.append(item['utterance'])
                #intent_labels.append(item['intent'])
                #print(item['utterance'], item['intent'])
        return lines


    def get_samples(self, labelled_examples, args, mode):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list,labels_list)

        return data

    def get_embedding(self, examples, label_list, args, mode="train"):
        sbert = SentenceTransformer(args.bert_model)
        tokenizer = sbert.tokenizer
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
        data = []
        for f in features:
            results = {
                "input_ids": f.input_ids,
                "input_mask": f.input_mask,
                "segment_ids": f.segment_ids,
                "label_id": f.label_id
            }

            data.append(results)

        return data

    def get_loader_v1(self, labelled_examples, label_list, args, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, args.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == 'train':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        else:
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

        return dataloader

    def get_loader_v2(self, examples, label_list, args, mode = 'train'):
        sbert = SentenceTransformer(args.bert_model)
        tokenizer = sbert.tokenizer

        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == 'train':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

        return dataloader


class OriginSamples(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_y) == len(train_x)
        self.train_x = train_x
        self.train_y = train_y


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    content_list = examples.train_x
    label_list = examples.train_y

    for i in range(len(content_list)):
        tokens_a = tokenizer.tokenize(content_list[i])

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        padding_2 = [1] * (max_seq_length - len(input_ids))
        input_ids += padding_2
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[label_list[i]]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features
