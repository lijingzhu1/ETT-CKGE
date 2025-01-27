from .BaseModel import *
def generate_mask(n, dim, mask_ratio):
    # Create random ratios between 0 and 1 for each dimension
    random_ratios = torch.rand((n, dim))
    
    # Create a mask where values are 1 if the ratio is greater than or equal to the mask_ratio, 0 otherwise
    mask = (random_ratios >= mask_ratio).float()
    
    return mask
class DLKGE(BaseModel):
    def __init__(self, args, kg) -> None:
        super(DLKGE, self).__init__(args, kg)
        self.old_triples_weights = []
        self.num_old_triples = self.args.num_old_triples
        self.num_old_entities = 1000
        self.degree_ent = {}
        self.degree_rel = {}
        self.new_degree_ent = {}
        self.new_degree_rel = {}
        self.mse_loss_func = nn.MSELoss(size_average=False)

    def pre_snapshot(self):
        """ propress before snapshot """
        if self.args.using_mask_weight and self.args.snapshot:
            self.num_new_entity = self.kg.snapshots[self.args.snapshot].num_ent - self.kg.snapshots[self.args.snapshot - 1].num_ent
            self.entity_weight_linear = nn.Linear(self.num_new_entity, self.num_new_entity, bias=False)
            constant_(self.entity_weight_linear.weight, 1e-3)
            self.entity_weight_linear.cuda()
            if self.args.using_MAE_loss:
                self.mask_weight_linear = nn.Sequential(
                    nn.Linear(self.args.emb_dim, self.args.emb_dim, bias=False).double(),
                    nn.ReLU()  # Using Tanh activation
                )
                constant_(self.mask_weight_linear[0].weight, 1e-3)
                self.mask_weight_linear.cuda()
            # self.num_entity = self.kg.snapshots[self.args.snapshot].num_ent
            # self.multi_layer_weight_linear = nn.Linear(self.num_entity, self.num_entity, bias=False)
            # constant_(self.multi_layer_weight_linear.weight, 1e-3)
            # self.multi_layer_weight_linear.cuda()


    def store_old_parameters(self):
        """ store last result """
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def store_previous_old_parameters(self):
        """ store previous results """
        # save_num = self.args.multi_distill_num # set store number
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(
                    f'old_data_{self.args.snapshot}_{name}', value.clone().detach()
                )

    def reply(self):
        print("using reply")
        """ ————————————————————————————————————————relpy for ratio———————————————————————————————————————————————— """
        self.old_triples_weights = list()
        i_sum = 0
        old_nums = []
        for i in range(0, self.args.snapshot + 1):
            i_sum += i + 1
        for i in range(0, self.args.snapshot + 1):
            old_nums.append((i + 1) * self.args.num_old_triples // i_sum)
        old_nums = old_nums[::-1]
        for i in range(len(old_nums)):
            self.old_triples_weights += list(random.sample(self.kg.snapshots[i].train, old_nums[i]))
        """ ————————————————————————————————————————relpy for ratio———————————————————————————————————————————————— """
        # """ ————————————————————————————————————————reply for number of a snapshot———————————————————————————————————————————————— """
        # self.old_triples_weights = list()
        # i_sum = 0
        # old_nums = []
        # for i in range(0, self.args.snapshot + 1):
        #     i_sum += len(self.kg.snapshots[i].train)
        # for i in range(0, self.args.snapshot + 1):
        #     old_nums.append(len(self.kg.snapshots[i].train) * self.args.num_old_triples // i_sum)
        # for i in range(len(old_nums)):
        #     self.old_triples_weights += list(random.sample(self.kg.snapshots[i].train, old_nums[i]))
        # """ ————————————————————————————————————————reply for number of a snapshot———————————————————————————————————————————————— """
        print(f"reply number{len(self.old_triples_weights)}")



    def switch_snapshot(self):
        if self.args.using_multi_embedding_distill == False:
            self.store_old_parameters() # save last embedding
        else:
            self.store_previous_old_parameters() # save previous embeddings
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = Parameter(
            self.ent_embeddings.weight.data
        )
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = Parameter(
            self.rel_embeddings.weight.data
        )
        self.ent_embeddings.weight = Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = Parameter(new_rel_embeddings)
        if self.args.using_structure_distill or self.args.using_score_distill or self.args.using_reply:
            self.reply()

    def embedding(self, stage=None):
        """ stage: Train, Valid, Test """
        if not self.args.use_two_stage or self.args.epoch > int(self.args.two_stage_epoch_num):
            return self.ent_embeddings.weight, self.rel_embeddings.weight
        else:
            new_ent_embeddings = self.ent_embeddings.weight
            new_rel_embeddings = self.rel_embeddings.weight
            if self.args.snapshot > 0:
                old_ent_embeddings = self.old_data_ent_embeddings_weight
                old_rel_embeddings = self.old_data_rel_embeddings_weight
                old_ent_len = self.kg.snapshots[self.args.snapshot - 1].num_ent
                old_rel_len = self.kg.snapshots[self.args.snapshot - 1].num_rel
                ent_embeddings = torch.cat([old_ent_embeddings[:old_ent_len], new_ent_embeddings[old_ent_len:]])
                rel_embeddings = torch.cat([old_rel_embeddings[:old_rel_len], new_rel_embeddings[old_rel_len:]])
            else:
                ent_embeddings = new_ent_embeddings
                rel_embeddings = new_rel_embeddings
            return ent_embeddings, rel_embeddings

class TransE(DLKGE):
    def __init__(self, args, kg) -> None:
        super(TransE, self).__init__(args, kg)
        self.huber_loss = torch.nn.HuberLoss(reduction='sum')

    def get_TransE_loss(self, head, relation, tail, label,first_layer_mask,second_layer_mask):
        return self.new_loss(head, relation, tail, label,first_layer_mask,second_layer_mask)

    def get_old_triples(self):
        if isinstance(self.old_triples_weights ,list):
            return self.old_triples_weights
        return list(self.old_triples_weights.keys())

    def structure_loss(self, triples):
        """ 计算结构相似度 """
        h = [x[0] for x in triples]
        h = torch.LongTensor(h).to(self.args.device)
        t = [x[2] for x in triples]
        t = torch.LongTensor(t).to(self.args.device)
        if self.args.using_multi_embedding_distill:
            old_ent_embeddings = getattr(
                self, f"old_data_{self.args.snapshot - 1}_ent_embeddings_weight"
            )
                # old_rel_embeddings = getattr(self, "old_data_{}_rel_embeddings_weight".format(self.args.snapshot - 1))
        else:
            old_ent_embeddings = self.old_data_ent_embeddings_weight
            # old_rel_embeddings = self.old_data_rel_embeddings_weight
        old_h = torch.index_select(old_ent_embeddings, 0, h)
        # old_r = torch.index_select(old_rel_embeddings, 0, r)
        old_t = torch.index_select(old_ent_embeddings, 0, t)
        new_h = torch.index_select(self.ent_embeddings.weight, 0, h)
        # new_r = torch.index_select(self.rel_embeddings, 0, r)
        new_t = torch.index_select(self.ent_embeddings.weight, 0, t)
        loss = self.huber_loss(F.cosine_similarity(old_h, old_t), F.cosine_similarity(new_h, new_t))
        old_h_t = torch.norm(old_h, dim=1) / torch.norm(old_t, dim=1)
        new_h_t = torch.norm(new_h, dim=1) / torch.norm(new_t, dim=1)
        loss += self.huber_loss(old_h_t, new_h_t)
        return loss

    def get_structure_distill_loss(self):
        if self.args.snapshot == 0:
            return 0.0
        """ 计算子图结构蒸馏 """
        triples = self.get_old_triples()
        return self.structure_loss(triples)

    def score_distill_loss(self, head, relation, tail):
        if self.args.using_multi_embedding_distill:
            old_ent_embeddings = getattr(
                self, f"old_data_{self.args.snapshot - 1}_ent_embeddings_weight"
            )
            old_rel_embeddings = getattr(
                self, f"old_data_{self.args.snapshot - 1}_rel_embeddings_weight"
            )
        else:
            old_ent_embeddings = self.old_data_ent_embeddings_weight
            old_rel_embeddings = self.old_data_rel_embeddings_weight
        new_h = torch.index_select(self.ent_embeddings.weight, 0, head)
        new_r = torch.index_select(self.rel_embeddings.weight, 0, relation)
        new_t = torch.index_select(self.ent_embeddings.weight, 0, tail)
        new_score = self.score_fun(new_h, new_r, new_t)
        old_h = torch.index_select(old_ent_embeddings, 0, head)
        old_r = torch.index_select(old_rel_embeddings, 0, relation)
        old_t = torch.index_select(old_ent_embeddings, 0, tail)
        old_score = self.score_fun(old_h, old_r, old_t)
        return self.huber_loss(old_score, new_score)

    def get_score_distill_loss(self):
        if self.args.snapshot == 0:
            return 0.0
        """ count subgraph score distillation """
        triples = self.get_old_triples()
        # triples, labels = self.corrupt(triples)
        triples = torch.LongTensor(triples).to(self.args.device)
        # labels = torch.Tensor(labels).to(self.args.device)
        head, relation, tail = triples[:, 0], triples[:, 1], triples[:, 2]
        return self.score_distill_loss(head, relation, tail)

    def corrupt(self, facts):
        '''
        Create negative samples by randomly corrupt subject or object entity
        :param triples:
        :return: negative samples
        '''
        ss_id = self.args.snapshot
        label = []
        facts_ = []
        prob = 0.5
        for fact in facts:
            s, r, o = fact[0], fact[1], fact[2]
            neg_s = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            neg_o = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            pos_s = np.ones_like(neg_s) * s
            pos_o = np.ones_like(neg_o) * o
            rand_prob = np.random.rand(self.args.neg_ratio)
            sub = np.where(rand_prob > prob, pos_s, neg_s)
            obj = np.where(rand_prob > prob, neg_o, pos_o)
            facts_.append((s, r, o))
            label.append(1)
            for ns, no in zip(sub, obj):
                facts_.append((ns, r, no))
                label.append(-1)
        return facts_, label

    def get_embedding_distillation_loss(self):
        """ count embedding distillation loss """
        if self.args.snapshot == 0:
            return 0.0
        losses = []
        for name, param in self.named_parameters():
            if name in ["snapshot_weights"]:
                continue
            name = name.replace('.', '_')
            old_data = getattr(self, f'old_data_{name}')
            new_data = param[:old_data.size(0)]
            assert new_data.size(0) == old_data.size(0)
            losses.append(self.huber_loss(old_data, new_data))
        return sum(losses)

    def get_one_layer_loss(self):
        """ count loss without distillation """
        if self.args.snapshot == 0:
            return 0.0
        loss = self.huber_loss(self.old_data_ent_embeddings_weight, self.ent_embeddings.weight)
        return loss


    def get_multi_layer_distance_loss(self, entity_mask, relation_mask, entity_mask_weight):
        """ count multy layer loss """
        # if not hasattr(self, 'old_layer_data_ent_embeddings_weight'):
        #     return 0.0
        if self.args.snapshot == 0:
            return 0.0
        else:
            if not hasattr(self, 'old_layer_data_ent_embeddings_weight'):
                return 0.0
            elif self.old_layer_data_ent_embeddings_weight.size(0) != self.ent_embeddings.weight.size(0) or (entity_mask is None):
                return 0.0 
            # new_entity_weight = self.multi_layer_weight_linear(entity_mask_weight)
            # print(f'entity_mask:',entity_mask.size())
            old_ent_embeddings = self.old_layer_data_ent_embeddings_weight * entity_mask.unsqueeze(1)*entity_mask_weight.unsqueeze(1)
            new_ent_embeddings = self.ent_embeddings.weight * entity_mask.unsqueeze(1)*entity_mask_weight.unsqueeze(1)
            loss = self.huber_loss(old_ent_embeddings, new_ent_embeddings)
            return loss
    # def get_real_multi_layer_loss(self,first_layer_mask,second_layer_mask):
    #     if self.args.epoch <= self.args.first_layer_epoch_num:
    #         return 0.0
    #     elif self.args.epoch <=self.args.second_layer_epoch_num and self.args.epoch >= self.args.first_layer_epoch_num:
    #         previous_layer_embeddings = self.first_layer_data_ent_embeddings_weight*first_layer_mask.unsqueeze(1)
    #         current_layer_embeddings = self.ent_embeddings.weight*first_layer_mask.unsqueeze(1)
    #         loss = self.huber_loss(previous_layer_embeddings, current_layer_embeddings)
    #         return loss
    #     elif self.args.epoch >self.args.second_layer_epoch_num:
    #         previous_layer_embeddings = self.second_layer_data_ent_embeddings_weight*second_layer_mask.unsqueeze(1)
    #         current_layer_embeddings = self.ent_embeddings.weight*second_layer_mask.unsqueeze(1)
    #         loss = self.huber_loss(previous_layer_embeddings, current_layer_embeddings)
    #         return loss
    def get_real_multi_layer_loss(self,first_layer_mask,second_layer_mask):
        if not 'first_layer_data_ent_embeddings_weight' in self._buffers:
            return 0.0
        if 'first_layer_data_ent_embeddings_weight' in self._buffers and not 'second_layer_data_ent_embeddings_weight' in self._buffers:
            # print(f'find it')
        # elif self.args.epoch <=self.second_layer_epoch_num and self.args.epoch >= self.args.first_layer_epoch_num:
            previous_layer_embeddings = self.first_layer_data_ent_embeddings_weight*first_layer_mask.unsqueeze(1)
            current_layer_embeddings = self.ent_embeddings.weight*first_layer_mask.unsqueeze(1)
            loss = self.huber_loss(previous_layer_embeddings, current_layer_embeddings)
            return loss
        elif 'second_layer_data_ent_embeddings_weight' in self._buffers:
            previous_layer_embeddings = self.second_layer_data_ent_embeddings_weight*second_layer_mask.unsqueeze(1)
            current_layer_embeddings = self.ent_embeddings.weight*second_layer_mask.unsqueeze(1)
            loss = self.huber_loss(previous_layer_embeddings, current_layer_embeddings)
            return loss

    def get_multi_layer_loss(self, entity_mask, relation_mask, entity_mask_weight):
        """ count multy layer loss """
        if self.args.snapshot == 0 or (self.args.use_two_stage and self.args.epoch < int(self.args.two_stage_epoch_num)):
            return 0.0
        if self.args.use_multi_layers and self.args.using_mask_weight:
            # print(entity_mask_weight.device)
            # print(entity_mask_weight.dtype)
            # entity_mask_weight[-self.num_new_entity:] = self.entity_weight_linear(entity_mask_weight[-self.num_new_entity:])
            new_entity_mask_weight = self.entity_weight_linear(entity_mask_weight[-self.num_new_entity:])
            # entity_mask_weight = F.sigmoid(entity_mask_weight)
            # entity_mask_weight = F.relu(entity_mask_weight)
            # print(entity_mask[self.kg.snapshots[self.args.snapshot - 1].num_ent:])
            entity_mask[-self.num_new_entity:] = entity_mask[-self.num_new_entity:].clone() * new_entity_mask_weight
            # entity_mask *= entity_mask_weight
            # print(entity_mask[self.kg.snapshots[self.args.snapshot - 1].num_ent:])
            # print(entity_mask_weight)
            # self.args.logger.info(entity_mask_weight)
        if self.args.using_mask_weight == False:
            entity_mask = torch.ones_like(entity_mask) * self.multi_layer_weight
        old_ent_embeddings = self.old_data_ent_embeddings_weight * entity_mask.unsqueeze(1)
        new_ent_embeddings = self.ent_embeddings.weight * entity_mask.unsqueeze(1)
        # new_ent_mask = generate_mask(new_ent_embeddings.size(0),self.args.emb_dim,self.args.mask_ratio).to(self.args.device).double()
        # new_ent_embeddings = new_ent_mask*new_ent_embeddings
        # new_ent_embeddings = self.mask_weight_linear(new_ent_embeddings) 
        # print(f'new_ent_embedidngs:',new_ent_embeddings.size())
        loss = self.huber_loss(old_ent_embeddings, new_ent_embeddings)
        # old_ent_len = self.kg.snapshots[self.args.snapshot - 1].num_ent
        # loss = self.huber_loss(old_ent_embeddings[old_ent_len:], new_ent_embeddings[old_ent_len:])
        # print(f'loss:',old_ent_embeddings[old_ent_len:].size(),loss)
        if self.args.using_relation_distill:
            old_rel_embeddings = self.old_data_rel_embeddings_weight * relation_mask.unsqueeze(1)
            new_rel_embeddings = self.rel_embeddings.weight * relation_mask.unsqueeze(1)
            loss += self.huber_loss(old_rel_embeddings, new_rel_embeddings)
        return loss

    def MAE_loss(self, entity_mask, relation_mask, entity_mask_weight):
        """ count multy layer loss """
        # if self.args.snapshot == 0 or (self.args.use_two_stage and self.args.epoch < int(self.args.two_stage_epoch_num)):
        if self.args.snapshot == 0:
            return 0.0
        old_ent_len = self.kg.snapshots[self.args.snapshot - 1].num_ent
        old_ent_embeddings = self.old_data_ent_embeddings_weight[:old_ent_len]
        new_ent_embeddings = self.ent_embeddings.weight[:old_ent_len]
        new_ent_mask = generate_mask(new_ent_embeddings.size(0),self.args.emb_dim,self.args.mask_ratio).to(self.args.device).double()
        new_ent_embeddings = new_ent_mask*new_ent_embeddings
        new_ent_embeddings = self.mask_weight_linear(new_ent_embeddings) 
        # new_ent_embeddings = new_ent_embeddings*entity_mask.unsqueeze(1)
        # old_ent_embeddings = old_ent_embeddings *entity_mask.unsqueeze(1)
        new_ent_embeddings = new_ent_embeddings*entity_mask[:old_ent_len].unsqueeze(1)
        old_ent_embeddings = old_ent_embeddings*entity_mask[:old_ent_len].unsqueeze(1)
        # print(f'new_ent_embedidngs:',new_ent_embeddings.size(0))
        # loss = self.mse_loss_func(old_ent_embeddings, new_ent_embeddings)
        loss = self.huber_loss(old_ent_embeddings, new_ent_embeddings)
        # if self.args.using_relation_distill:
        #     old_rel_embeddings = self.old_data_rel_embeddings_weight * relation_mask.unsqueeze(1)
        #     new_rel_embeddings = self.rel_embeddings.weight * relation_mask.unsqueeze(1)
        #     loss += self.huber_loss(old_rel_embeddings, new_rel_embeddings)
        return loss

    def get_multi_embedding_distillation_loss(self):
        """ count multylayer loss """
        if self.args.snapshot == 0:
            return 0.0
        losses = []
        for name, param in self.named_parameters():
            if name == "snapshot_weights":
                continue
            name = name.replace('.', '_')
            for i in range(self.args.snapshot):
                old_data = getattr(self, f'old_data_{i}_{name}')
                new_data = param[:old_data.size(0)]
                assert new_data.size(0) == old_data.size(0)
                losses.append(self.huber_loss(old_data, new_data))
        s_weights = self.snapshot_weights.to(self.args.device).double()
        weights_softmax = F.softmax(s_weights, dim=-1)
        losses = torch.cat([loss.unsqueeze(0) for loss in losses], dim=0)
        loss = torch.dot(losses, weights_softmax)
        print(self.snapshot_weights.grad)
        print(self.snapshot_weights)
        return loss


    def get_reply_loss(self, new_triples, new_labels):
        if self.args.snapshot == 0:
            return 0.0
        """ count subgraph score distillation """
        old_triples = self.get_old_triples()
        old_triples, old_labels = self.corrupt(old_triples)
        old_triples = torch.LongTensor(old_triples).to(self.args.device)
        old_labels = torch.Tensor(old_labels).to(self.args.device)
        new_triples = torch.cat([new_triples, old_triples], dim=0)
        new_labels = torch.cat([new_labels, old_labels], dim=0)
        head, relation, tail = new_triples[:, 0], new_triples[:, 1], new_triples[:, 2]
        return self.new_loss(head, relation, tail, new_labels)

    def get_contrast_loss(self):
        if self.args.snapshot == 0:
            return 0.0
        old_ent_embeds = self.old_data_ent_embeddings_weight
        old_rel_embeds = self.old_data_rel_embeddings_weight
        new_ent_embeds = self.ent_embeddings.weight
        new_rel_embeds = self.rel_embeddings.weight
        losses = []
        idxs = set()
        for ent in self.new_degree_ent:
            if ent < old_ent_embeds.size(0):
                idxs.add(ent)
        # print(len(idxs))
        for idx in idxs:
            all_poses = []
            all_poses.append(idx)
            neg_poses = random.sample(range(old_ent_embeds.size(0)), self.args.neg_ratio - 5)
            while idx in neg_poses:
                neg_poses = random.sample(range(old_ent_embeds.size(0)), self.args.neg_ratio - 5)
            all_poses += neg_poses
            student_ent_embeds = new_ent_embeds[all_poses]
            teacher_ent_embeds = old_ent_embeds[all_poses]
            losses.append(infoNCE(student_ent_embeds, teacher_ent_embeds, [0]))
        return sum(losses)

    def loss(self, head, relation, tail=None, label=None, entity_mask=None, relation_mask=None, entity_mask_weight=None,first_layer_mask=None,second_layer_mask=None):
        loss = 0.0;decompose_loss = 0.0;MAE_loss = 0.0
        """ 0. count initial loss """
        if not self.args.using_reply or self.args.snapshot == 0:
            transE_loss = self.get_TransE_loss(head, relation, tail, label,first_layer_mask,second_layer_mask) # 本次snapshot的TransE损失
            # loss = transE_loss
            loss += transE_loss
        # """ 1. embedding distill loss """
        # if self.args.using_embedding_distill:
        #     embedding_distillation_loss = self.get_embedding_distillation_loss()
        #     loss += embedding_distillation_loss * self.args.embedding_distill_weight
        # if self.args.without_multi_layers:
        #     one_layer_loss = self.get_one_layer_loss() * self.args.embedding_distill_weight
        #     loss += one_layer_loss
        """ 1. incremental distillation """
        # snapshot_loss_weights = [0.01, 0.1, 1, 1]
        # MAE_loss_weights = [0.001, 0.001, 0.001, 0.001]
        # MAE_loss_weights = [0.0005, 0.0005, 0.0005, 0.0001] #best performance
        # MAE_loss_weights = [0.0001, 0.0001, 0.0001, 0.0001]
        MAE_loss_weights = self.args.MAE_loss_weights
        if self.args.snapshot == 0:
            MAE_loss_weight = 0
        if self.args.snapshot == 1:
            MAE_loss_weight = MAE_loss_weights[0]
        elif self.args.snapshot == 2:
            MAE_loss_weight = MAE_loss_weights[1]
        elif self.args.snapshot == 3:
            MAE_loss_weight = MAE_loss_weights[2]
        elif self.args.snapshot == 4:
            MAE_loss_weight = MAE_loss_weights[3] 
        if self.args.use_multi_layers and (not self.args.without_multi_layers):
            # print(f'using multi layers')
            multi_layer_loss = self.get_multi_layer_loss(entity_mask, relation_mask, entity_mask_weight)* float(self.args.multi_layer_weight)
            loss += multi_layer_loss 
        if self.args.using_MAE_loss:
            MAE_loss = self.MAE_loss(entity_mask, relation_mask, entity_mask_weight)* float(MAE_loss_weight)
            loss += MAE_loss 
        if self.args.using_multi_layer_distance_loss:
            # decompose_loss = self.get_multi_layer_distance_loss(last_entity_mask, relation_mask, entity_mask_weight)* float(self.args.multi_layer_distance_weight)
            # loss += decompose_loss
            decompose_loss = self.get_real_multi_layer_loss(first_layer_mask,second_layer_mask)* float(self.args.multi_layer_distance_weight)
            loss += decompose_loss
            # print(f'loss after MAE_loss:',loss)
        # print(f'transE_loss,multi_layer_loss,MAE_loss,loss:',transE_loss,multi_layer_loss,MAE_loss,loss)
        return transE_loss,multi_layer_loss,MAE_loss,decompose_loss,loss

