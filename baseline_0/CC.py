import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from contrastive_loss import *

class CCModelManager:

    def __init__(self, args, data, pretrained_model=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_labels = data.n_unknown_cls
        print("novel_num_label", self.num_labels)

        self.model = BertForModel.from_pretrained(args.bert_model, num_labels=self.num_labels)

        self.model.to(self.device)

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)

        print(self.model)

        num_train_examples = len(data.train_unlabeled_examples.train_x)
        print("num_OOD_train_examples:",num_train_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs

        self.optimizer = self.get_optimizer(args)

        self.criterion_instance = InstanceLoss(args.train_batch_size, args.instance_temperature, self.device).to(
            self.device)
        self.criterion_cluster = ClusterLoss(self.num_labels, args.cluster_temperature, self.device).to(
            self.device)

        print(self.model)

        self.best_eval_score = 0
        self.centroids = None

        self.test_results = None
        self.training_SC_epochs = {}
        self.test_results_SC_list = {}
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

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        return optimizer

    def pca_visualization(self,x,y,predicted, args):
        label_list=[0,1,2,3,4,5,6,7,8,9]
        path = args.save_results_path
        pca_visualization(x, y, label_list, os.path.join(path, "pca_test.png"))
        pca_visualization(x, predicted, label_list, os.path.join(path, "pca_test_2.png"))

    def tsne_visualization(self,x,y,predicted, args):
        label_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        path = args.save_results_path
        TSNE_visualization(x, y, label_list, os.path.join(path, "pca_test_b2.png"))
        TSNE_visualization(x, predicted, label_list, os.path.join(path, "pca_test_2_b2.png"))

    def tsne_visualization_2(self,x,y,predicted, args, epoch=100):
        label_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        path = args.save_results_path
        TSNE_visualization(x, y, label_list, os.path.join(path, "pca_train_"+str(args.seed)+str(epoch)+".pdf"))



    def evaluation(self, args, data):

        self.model.eval()
        eval_dataloader = data.test_unlabeled_dataloader
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch, pretrain=False)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = total_logits.max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results = clustering_score(y_true, y_pred)
        print(results)
        self.test_results = results

        return results
        # return feature_vector, labels_vector

    def visualize_training(self, args, data):
        self.model.eval()
        eval_dataloader = data.train_unlabeled_dataloader
        total_features = torch.empty((0, 768)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = total_logits.max(dim=1)
        total_features = normalize(total_features, dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        results = clustering_score(y_true, y_pred)
        print(results)
        self.train_results = results

        # self.pca_visualization(x_feats, y_true, y_pred)
        self.tsne_visualization_2(x_feats, y_true, y_pred, args)

    def eval(self, args, data):
        self.model.eval()
        eval_dataloader = data.eval_unlabeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch, pretrain=False)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, feat))

        total_probs, total_preds = total_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        score = metrics.silhouette_score(x_feats, y_pred)

        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results = clustering_score(y_true, y_pred)
        #print(results)
        #self.test_results = results

        return score


    def training_process_eval(self, args, data, epoch):
        self.model.eval()
        eval_dataloader = data.train_unlabeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, feat))

        total_probs, total_preds = total_logits.max(dim=1)
        total_features = normalize(total_features, dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        results = clustering_score(y_true, y_pred)
        score = metrics.silhouette_score(x_feats, y_pred)
        #score = results["NMI"]
        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        self.training_SC_epochs["epoch:" + str(epoch)] = score

        self.tsne_visualization_2(x_feats, y_true, y_pred, epoch)

        return score


    def train(self, args, data):

        best_score = 0
        best_model = None
        wait = 0
        e_step = 0


        train_dataloader_1 = data.train_unlabeled_dataloader

        #eval_acc = self.eval(args, data)
        #print(eval_acc)
        #self.training_SC_epochs["epoch:" + str(e_step)] = eval_acc
        #e_step+=1

        # contrastive clustering
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            step = 0
            loss_epoch = 0
            for step, batch in enumerate(tqdm(data.train_unlabeled_dataloader, desc="Pseudo-Training")):
                batch = tuple(t.to(self.device) for t in batch)

                input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = batch

                z_i, z_j, c_i, c_j = self.model(batch, mode='contrastive-clustering', pretrain=False)

                loss_instance = self.criterion_instance(z_i, z_j)
                loss_cluster = self.criterion_cluster(c_i, c_j)
                loss = loss_instance + loss_cluster
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
                print(
                    f"Step [{step}/{len(train_dataloader_1)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
                loss_epoch += loss.item()
            print(f"Epoch [{epoch}/{args.num_train_epochs}]\t Loss: {loss_epoch / len(train_dataloader_1)}")

            eval_acc = self.eval(args, data)
            print(eval_acc)
            #self.training_SC_epochs["epoch:" + str(e_step)] = eval_acc
            #e_step += 1

            if eval_acc > best_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_score = eval_acc
            else:
                wait += 1
                if wait >= args.wait_patient:
                    self.model = best_model
                    break

    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias', 'cluster_projector.2.weight',
                             'cluster_projector.2.bias', 'cluster_projector.4.weight', 'cluster_projector.4.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'cluster_num_factor', 'seed', 'K']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results_no_1.csv'
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
        #self.save_training_process(args)

    def save_training_process(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        results = dict(self.training_SC_epochs)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results_analysis_V1_1_trainigEpoch.csv'
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

        print('training_process_dynamic:', data_diagram)
