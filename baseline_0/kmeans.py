from util import *
from pretrain import *
from torch.nn.functional import normalize

class KmeansModelManager:

    def __init__(self, args, data, pretrained_model=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        set_seed(args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pretrained_model is None:
            pretrained_model = BertForModel.from_pretrained(args.bert_model, num_labels = data.n_known_cls)
            pretrained_model.to(self.device)
            root_path = "pretrain_models"
            pretrain_dir = os.path.join(root_path, args.pretrain_dir)
            if os.path.exists(pretrain_dir):
                pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model

        self.pretrained_model.to(self.device)

        self.num_labels = data.n_unknown_cls
        #self.num_labels = self.predict_k(args, data)
        print("novel_num_label", self.num_labels)
        self.model = self.pretrained_model

        if args.pretrain:
            self.load_pretrained_model(args)

        self.model.to(self.device)

        #num_train_examples = len(data.train_unlabeled_examples.train_x1)

        self.best_eval_score = 0
        self.centroids = None

        self.test_results = {}
        self.predictions = None
        self.true_labels = None

    def load_models(self, args):
        print("loading models ....")
        self.model = self.restore_model_v2(args, self.model)

    def get_features_labels(self, dataloader, model, args):

        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(batch, mode='feature-extract', pretrain = True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data):

        feats, _ = self.get_features_labels(data.train_unlabeled_dataloader, self.pretrained_model, args)
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


        #self.pca_visualization(x_feats, y_true, y_pred)

        #file = "./outputs/results.csv"
        #with open(file, "w") as f:
        #    f.write(results)
        #    f.write("ground_truth\t")
        #    f.write(y_true)
        #    f.write("prediction\t")
        #    f.write(y_pred)
        #f.close()
        #print(y_true)
        #print(len(y_pred[y_pred>0.5]))

        self.test_results = results

        return results


    '''
    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias', 'cluster_projector.2.weight',
                             'cluster_projector.2.bias', 'cluster_projector.4.weight', 'cluster_projector.4.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
    '''

    def restore_model(self, args, model):
        root_path = "pretrain_models"
        pretrain_dir = os.path.join(root_path, args.pretrain_dir)
        output_model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias', 'cluster_projector.2.weight', 'cluster_projector.2.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def restore_model_v2(self, args, model):
        root_path = "pretrain_models"
        pretrain_dir = os.path.join(root_path, args.pretrain_dir)
        output_model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.cluster_num_factor, args.seed,
               args.train_batch_size, args.lr, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'cluster_num_factor', 'seed', 'train_batch_size',
                 'learning_rate', 'K']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results_kmeans_1.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)

    def save_results_2(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.cluster_num_factor, args.seed,
               args.train_batch_size, args.lr, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'cluster_num_factor', 'seed', 'train_batch_size',
                 'learning_rate', 'K']
        vars_dict = {k: v for k, v in zip(names, var)}

        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results_analysis_V4_1_intra_inter.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)


    def BertForKmeans(self, args, data):
        feats, labels = self.get_features_labels(data.train_unlabeled_dataloader, self.model, args)
        feats = normalize(feats, dim=1)
        feats = feats.cpu().numpy()

        km = KMeans(n_clusters=self.num_labels).fit(feats)
        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred)
        print('results', results)

        #ind, _ = hungray_aligment(y_true, y_pred)
        #map_ = {i[0]: i[1] for i in ind}
        #y_pred = np.array([map_[idx] for idx in y_pred])

        #cm = confusion_matrix(y_true, y_pred)
        score = metrics.silhouette_score(feats, y_pred)
        #print('confusion matrix', cm)
        self.test_results = results
        self.test_results["SC"] = score

        min_d, max_d, mean_d, list_1 = intra_distance(feats, y_true, self.num_labels)
        self.test_results["intra_distance"] = mean_d
        min_d, max_d, mean_d, list_2 = inter_distance(feats, y_true, self.num_labels)
        self.test_results["inter_distance"] = mean_d

        print(self.test_results)

    def analysis(self, args, data):
        feats, labels = self.get_features_labels(data.test_unlabeled_dataloader, self.model, args)
        feats = normalize(feats, dim=1)
        feats = feats.cpu().numpy()
        y_true = labels.cpu().numpy()

        km = KMeans(n_clusters=self.num_labels).fit(feats)
        y_pred = km.labels_

        score = metrics.silhouette_score(feats, y_true)
        self.test_results["SC"] = score
        min_d, max_d, mean_d = intra_distance(feats, y_true, self.num_labels)
        self.test_results["intra_distance"] = mean_d
        min_d, max_d, mean_d = inter_distance(feats, y_true, self.num_labels)
        self.test_results["inter_distance"] = mean_d

        print(self.test_results)