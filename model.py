from util import *
import torch.nn as nn
import torch
from torch.nn.functional import normalize
from torch.nn import Parameter
import math
from keras.utils.np_utils import to_categorical
from contrastive_loss import *


def onehot_labelling(int_labels, num_classes):
    categorical_labels = to_categorical(int_labels, num_classes=num_classes)
    return categorical_labels

def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
    #print(x.shape)
    #print(x_adv.shape)
    #print(n.shape)
    #print(n_adv.shape)
    #print((n * n.t()).shape)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

def nt_xent(x, x_adv, mask, cuda=True, t=0.5):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).long()

    if cuda:
        dis = (x * (mask - torch.eye(x.size(0)).long().cuda()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().cuda()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    else:
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse

    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    #loss = dis.sum(1) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + dis_adv.sum(1) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -loss.mean()
    #return -torch.log(loss).mean()



class BertForModel(BertPreTrainedModel):
    def __init__(self,config, num_labels):
        super(BertForModel, self).__init__(config)

        self.num_labels = num_labels

        self.bert = BertModel(config) # 这个是backbone
        self.rnn = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=1,
                          dropout=config.hidden_dropout_prob, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 以上为编码器pooling层
        self.instance_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 128),
        ) # instance-level 投影

        self.cluster_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels),
        ) # class(cluster)-level 投影

        self.softmax = nn.Softmax(dim=1)

        self.apply(self.init_bert_weights)


    def forward(self, batch1 = None, mode = None, pretrain = True, positive_sample=None, negative_sample=None):
        if pretrain:
            if mode == "pre-trained":
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output = self.rnn(encoded_layer_12)
                pooled_output = torch.cat((pooled_output[0].squeeze(0),pooled_output[1].squeeze(0)),dim=1)
                pooled_output = self.dense(pooled_output)
                pooled_output = self.activation(pooled_output)
                pooled_output = self.dropout(pooled_output)

                # Class-level 损失函数
                logits = self.cluster_projector(pooled_output)
                ce_loss = nn.CrossEntropyLoss()(logits, label_ids)


                # Instance-level 损失函数
                z_i = self.instance_projector(pooled_output)

                label_ids = label_ids.cpu()
                labels = onehot_labelling(label_ids, self.num_labels)
                labels = torch.from_numpy(labels)
                labels = labels.cuda()
                label_mask = torch.mm(labels, labels.T).bool().long()
                encoded_layer_12_02, pooled_output_02 = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output_02 = self.rnn(encoded_layer_12_02)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)
                pooled_output_02 = self.dense(pooled_output_02)
                pooled_output_02 = self.activation(pooled_output_02)
                pooled_output_02 = self.dropout(pooled_output_02)
                z_j = self.instance_projector(pooled_output_02)
                sup_cont_loss = nt_xent(z_i, z_j, label_mask, cuda=True)

                loss = ce_loss + sup_cont_loss

                return loss

            elif mode == "contrastive-clustering":
                input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = batch1

                encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)
                encoded_layer_12_emb02, pooled_output_02 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)

                _, pooled_output_01 = self.rnn(encoded_layer_12_emb01)
                _, pooled_output_02 = self.rnn(encoded_layer_12_emb02)

                pooled_output_01 = torch.cat((pooled_output_01[0].squeeze(0), pooled_output_01[1].squeeze(0)), dim=1)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)

                pooled_output_01 = self.dense(pooled_output_01)
                pooled_output_02 = self.dense(pooled_output_02)

                pooled_output_01 = self.activation(pooled_output_01)
                pooled_output_02 = self.activation(pooled_output_02)

                pooled_output_01 = self.dropout(pooled_output_01)
                pooled_output_02 = self.dropout(pooled_output_02)

                z_i = normalize(self.instance_projector(pooled_output_01), dim=1)
                z_j = normalize(self.instance_projector(pooled_output_02), dim=1)

                c_i = self.cluster_projector(pooled_output_01)
                c_j = self.cluster_projector(pooled_output_02)

                #c_i = self.classifier(c_i)
                #c_j = self.classifier(c_j)

                c_i = self.softmax(c_i)
                c_j = self.softmax(c_j)

                return z_i, z_j, c_i, c_j

            elif mode == "feature-extract":
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                #_, pooled_output = self.rnn(encoded_layer_12)
                #pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
                pooled_output = encoded_layer_12.mean(dim = 1)

                return pooled_output

            else:
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output = self.rnn(encoded_layer_12)
                pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
                pooled_output = self.dense(pooled_output)
                pooled_output = self.activation(pooled_output)
                pooled_output = self.dropout(pooled_output)

                logits = self.cluster_projector(pooled_output)

                feats = normalize(pooled_output, dim=1)

                return feats, logits


    def forward_cluster(self, batch, pretrain = True):
        if pretrain:
            input_ids, input_mask, segment_ids, label_ids = batch
            encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                        output_all_encoded_layers=False)
            _, pooled_output = self.rnn(encoded_layer_12)
            pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)

            c = self.cluster_projector(pooled_output)
            c = self.softmax(c)

            return c, pooled_output


class MPnetForModel(nn.Module):
    def __init__(self, bert_model, num_labels, params=None):
        super(MPnetForModel, self).__init__()
        '''
        self.params = params
        self.K = self.params["queue_size"]
        self.top_k = self.params["top_k"]
        self.end_k = self.params["end_k"]
        self.update_num = self.params["positive_num"]
        self.contrastive_rate_in_training = self.params["contrastive_rate"]
        self.memory_bank = self.params["memory_bank"]
        self.T = 0.5  # 温度系数
        self.eta = 5
        '''

        self.num_labels = num_labels
        hidden_size = 768
        hidden_dropout_prob = 0.1
        self.sentbert = bert_model[0].auto_model
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1,
                          dropout=hidden_dropout_prob, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        #self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(hidden_dropout_prob)  # 以上为编码器pooling层
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.instance_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
        )  # instance-level 投影

        self.cluster_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_labels),
        )  # class(cluster)-level 投影

        self.softmax = nn.Softmax(dim=1)
        '''
        self.register_buffer("label_queue", torch.randint(0, self.num_labels, [self.K]))
        self.register_buffer("feature_queue", torch.randn(self.K, 768))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)
        '''

    '''
    def _dequeue_and_enqueue(self, keys, label):
        # TODO 我们训练过程batch_size是一个变动的，每个epoch的最后一个batch数目后比较少，这里需要进一步修改
        # keys = concat_all_gather(keys)
        # label = concat_all_gather(label)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            head_size = self.K - ptr
            head_keys = keys[: head_size]
            head_label = label[: head_size]
            end_size = ptr + batch_size - self.K
            end_key = keys[head_size:]
            end_label = label[head_size:]
            self.feature_queue[ptr:, :] = head_keys
            self.label_queue[ptr:] = head_label
            self.feature_queue[:end_size, :] = end_key
            self.label_queue[:end_size] = end_label
        else:
            # replace the keys at ptr (dequeue ans enqueue)
            self.feature_queue[ptr: ptr + batch_size, :] = keys
            self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr


    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def select_pos_neg_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()        # K
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        #print(feature_queue.shape, label_queue.shape)

        # 1、将label_queue和feature_queue扩展到batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size
        #print(tmp_feature_queue.shape, tmp_label_queue.shape)

        # 2、计算相似度
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])
        #print(cos_sim)
        #print(cos_sim.shape)

        # 3、根据label取正样本和负样本的mask_index
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])



        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index
        #print(pos_mask_index)
        #print(pos_mask_index.shape)


        #print(pos_mask_index.shape, neg_mask_index.shape)
        #print("------------------------------------")
        #print(neg_mask_index)


        # 4、根据mask_index取正样本和负样本的值
        feature_value = cos_sim.masked_select(pos_mask_index)
        #print(feature_value.shape)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        #print(pos_sample.shape)
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)
        #print(pos_sample.shape)
        #print("------------------------------------")

        feature_value = cos_sim.masked_select(neg_mask_index)
        #print(feature_value.shape)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        #print(neg_sample.shape)
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)
        #print(neg_sample.shape)

        #print("##############################")
        #print(pos_sample.shape, neg_sample.shape)

        # 5、取所有的负样本和前top_k 个正样本， -M个正样本（离中心点最远的样本）
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = pos_sample.topk(self.top_k, dim=-1)
        #print(pos_sample.shape)
        pos_sample_top_k = pos_sample[:, 0:self.top_k]
        #print(pos_sample_top_k.shape)
        #exit()

        #pos_sample_last = pos_sample[:, -self.end_k:]
        # pos_sample_last = pos_sample_last.view([-1, 1])

        #pos_sample = pos_sample_top_k
        #print(pos_sample.shape)
        #pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample_top_k.contiguous().view([-1, 1])
        #print(pos_sample.shape)

        #print("##############################")
        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        #print(neg_sample.shape)
        neg_sample = neg_sample.repeat([1, self.top_k])
        #print(neg_sample.shape)
        neg_sample = neg_sample.view([-1, neg_min])
        #print(neg_sample.shape)

        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        #print(logits_con.shape)

        logits_con /= self.T

        return logits_con
    '''

    def update(self, num_labels, cluster_centers):
        self.num_labels = num_labels
        hidden_size = 768
        self.cluster_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_labels),
        )  # class(cluster)-level 投影

        self.alpha = 1.0
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)
        print(cluster_centers)

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def forward(self, batch1 = None, mode = None, pretrain = True, positive_sample=None, negative_sample=None):
        if pretrain:
            if mode == "pre-trained":
                input_ids, input_mask, segment_ids, label_ids = batch1
                bert_output = self.sentbert.forward(input_ids, input_mask)
                all_output = bert_output[0]

                _, pooled_output = self.rnn(all_output)
                pooled_output = torch.cat((pooled_output[0].squeeze(0),pooled_output[1].squeeze(0)),dim=1)
                pooled_output = self.dense(pooled_output)
                #pooled_output = self.activation(pooled_output)
                #pooled_output = self.dropout(pooled_output)

                # Class-level 损失函数
                logits = self.cluster_projector(pooled_output)
                ce_loss = nn.CrossEntropyLoss()(logits, label_ids)
                #logits = self.classifier(pooled_output)
                #ce_loss = nn.CrossEntropyLoss()(logits, label_ids)

                # Instance-level 损失函数
                z_i = self.instance_projector(pooled_output)

                label_ids = label_ids.cpu()
                labels = onehot_labelling(label_ids, self.num_labels)
                labels = torch.from_numpy(labels)
                labels = labels.cuda()
                label_mask = torch.mm(labels, labels.T).bool().long()
                bert_output_02 = self.sentbert.forward(input_ids, input_mask)
                all_output_02 = bert_output_02[0]
                _, pooled_output_02 = self.rnn(all_output_02)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)
                pooled_output_02 = self.dense(pooled_output_02)
                #pooled_output_02 = self.activation(pooled_output_02)
                #pooled_output_02 = self.dropout(pooled_output_02)
                z_j = self.instance_projector(pooled_output_02)
                sup_cont_loss = nt_xent(z_i, z_j, label_mask, cuda=True)

                loss = ce_loss + sup_cont_loss
                #print(ce_loss, sup_cont_loss)

                '''
                loss = ce_loss
                
                if self.memory_bank:
                    with torch.no_grad():
                        #self.update_encoder_k()
                        update_sample = self.reshape_dict(positive_sample)
                        #print(positive_sample["input_ids"].shape, positive_sample["input_mask"].shape, positive_sample["segment_ids"].shape)
                        #exit()

                        bert_output_k = self.sentbert.forward(update_sample["input_ids"], update_sample["input_mask"])
                        all_output_k = bert_output_k[0]
                        _, pooled_output_k = self.rnn(all_output_k)
                        pooled_output_k = torch.cat((pooled_output_k[0].squeeze(0), pooled_output_k[1].squeeze(0)),
                                                    dim=1)
                        pooled_output_k = self.dense(pooled_output_k)
                        pooled_output_k = self.activation(pooled_output_k)
                        pooled_output_k = self.dropout(pooled_output_k)
                        #update_keys = normalize(pooled_output_k, dim=1)
                        update_keys = pooled_output_k


                        tmp_labels = label_ids.unsqueeze(-1)
                        tmp_labels = tmp_labels.repeat([1, self.update_num])
                        tmp_labels = tmp_labels.view(-1)
                        #print(tmp_labels, tmp_labels.shape)
                        #exit()
                        self._dequeue_and_enqueue(update_keys, tmp_labels)

                    #z_i = normalize(pooled_output, dim=1)
                    #print(pooled_output)
                    #exit()
                    logits_con = self.select_pos_neg_sample(pooled_output, label_ids)

                    if logits_con is not None:
                        labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
                        loss_con = nn.CrossEntropyLoss()(logits_con, labels_con)
                        #print("loss_con:",loss_con)
                        # loss = loss_con
                        loss = loss_con * self.contrastive_rate_in_training + \
                               ce_loss * (1 - self.contrastive_rate_in_training)
                '''
                return loss

            elif mode == "contrastive-clustering":
                input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = batch1

                bert_output_01 = self.sentbert.forward(input_ids_1, input_mask_1)
                all_output_01 = bert_output_01[0]
                bert_output_02 = self.sentbert.forward(input_ids_1, input_mask_1)
                all_output_02 = bert_output_02[0]

                _, pooled_output_01 = self.rnn(all_output_01)
                _, pooled_output_02 = self.rnn(all_output_02)

                pooled_output_01 = torch.cat((pooled_output_01[0].squeeze(0), pooled_output_01[1].squeeze(0)), dim=1)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)

                pooled_output_01 = self.dense(pooled_output_01)
                pooled_output_02 = self.dense(pooled_output_02)

                pooled_output_01 = self.activation(pooled_output_01)
                pooled_output_02 = self.activation(pooled_output_02)

                pooled_output_01 = self.dropout(pooled_output_01)
                pooled_output_02 = self.dropout(pooled_output_02)

                z_i = normalize(self.instance_projector(pooled_output_01), dim=1)
                z_j = normalize(self.instance_projector(pooled_output_02), dim=1)

                c_i = self.cluster_projector(pooled_output_01)
                c_j = self.cluster_projector(pooled_output_02)

                #c_i = self.classifier(c_i)
                #c_j = self.classifier(c_j)

                c_i = self.softmax(c_i)
                c_j = self.softmax(c_j)

                return z_i, z_j, c_i, c_j

            elif mode == "SCCL":
                input_ids, input_mask, segment_ids, label_ids = batch1
                bert_output_01 = self.sentbert.forward(input_ids, input_mask)
                all_output_01 = bert_output_01[0]
                bert_output_02 = self.sentbert.forward(input_ids, input_mask)
                all_output_02 = bert_output_02[0]

                attention_mask = input_mask.unsqueeze(-1)
                mean_output_01 = torch.sum(all_output_01 * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
                mean_output_02 = torch.sum(all_output_02 * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

                cluster_output_01 = self.get_cluster_prob(mean_output_01)
                cluster_output_02 = self.get_cluster_prob(mean_output_02)

                z_i = normalize(self.instance_projector(mean_output_01), dim=1)
                z_j = normalize(self.instance_projector(mean_output_02), dim=1)

                return z_i, z_j, cluster_output_01, cluster_output_02


            elif mode == "feature-extract":
                input_ids, input_mask, segment_ids, label_ids = batch1
                bert_output = self.sentbert.forward(input_ids, input_mask)
                all_output = bert_output[0]
                attention_mask = input_mask.unsqueeze(-1)
                mean_output = torch.sum(all_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
                #_, pooled_output = self.rnn(all_output)
                #pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
                #pooled_output = self.dense(pooled_output)
                pooled_output = mean_output

                return pooled_output

            else:
                input_ids, input_mask, segment_ids, label_ids = batch1
                bert_output = self.sentbert.forward(input_ids, input_mask)
                all_output = bert_output[0]
                _, pooled_output = self.rnn(all_output)
                pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
                pooled_output = self.dense(pooled_output)
                pooled_output = self.activation(pooled_output)
                pooled_output = self.dropout(pooled_output)

                logits = self.cluster_projector(pooled_output)

                feats = normalize(pooled_output, dim=1)

                return feats, logits


    def forward_cluster(self, batch, pretrain = True):
        if pretrain:
            input_ids, input_mask, segment_ids, label_ids = batch
            bert_output = self.sentbert.forward(input_ids, input_mask)
            all_output = bert_output[0]

            #attention_mask = input_mask.unsqueeze(-1)
            #mean_output = torch.sum(all_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            #cluster_output = self.get_cluster_prob(mean_output)

            _, pooled_output = self.rnn(all_output)
            pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
            pooled_output = self.dense(pooled_output)

            #pooled_output = self.dense_2(mean_output)
            #pooled_output = self.activation(pooled_output)
            #pooled_output = self.dropout(pooled_output)

            c = self.cluster_projector(pooled_output)
            c = self.softmax(c)

            return c, pooled_output
        else:
            input_ids, input_mask, segment_ids, label_ids = batch
            bert_output = self.sentbert.forward(input_ids, input_mask)
            all_output = bert_output[0]

            attention_mask = input_mask.unsqueeze(-1)
            mean_output = torch.sum(all_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            cluster_output = self.get_cluster_prob(mean_output)
            '''
            _, pooled_output = self.rnn(all_output)
            pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)

            c = self.cluster_projector(pooled_output)
            c = self.softmax(c)
            '''
            return cluster_output, mean_output

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()