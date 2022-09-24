from util import *
from pretrain import *

class KmeansModelManager:

    def __init__(self, args, data, pretrained_model=None):
        '''
        if pretrained_model is None:
            pretrained_model = BertForModel.from_pretrained(args.bert_model, num_labels=data.n_known_cls)
            if os.path.exists(args.pretrain_dir):
                pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model
        '''

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.pretrained_model.to(self.device)

        self.num_labels = self.predict_k(args, data)

        print("novel_num_label", self.num_labels)
        self.model = BertForModel.from_pretrained(args.bert_model, num_labels=self.num_labels)

        #if args.pretrain:
        #    self.load_pretrained_model(args)

        self.model.to(self.device)

        #num_train_examples = len(data.train_unlabeled_examples.train_x1)

        self.best_eval_score = 0
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def get_features_labels(self, dataloader, model, args):

        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(batch, mode='feature-extract')

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data):

        feats, _ = self.get_features_labels(data.train_unlabeled_dataloader_1, self.pretrained_model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels
        print('drop', drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label])
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num', num_labels)

        return num_labels

    '''
    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias', 'cluster_projector.2.weight',
                             'cluster_projector.2.bias', 'cluster_projector.4.weight', 'cluster_projector.4.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
    '''


    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model


    def BertForKmeans(self, args, data):
        feats, _ = self.get_features_labels(data.train_unlabeled_dataloader_1, self.pretrained_model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=data.num_labels).fit(feats)
        y_pred = km.labels_