from .tokened_BaseModel import *

class DLKGE(Tokened_BaseModel):
    def __init__(self, args, kg) -> None:
        super(DLKGE, self).__init__(args, kg)
        self.init_old_weight()
        self.old_triples_weights = []
        self.num_old_triples = self.args.num_old_triples
        self.num_old_entities = 1000
        self.degree_ent = {}
        self.degree_rel = {}
        self.new_degree_ent = {}
        self.new_degree_rel = {}
        self.mse_loss_func = nn.MSELoss(size_average=False)

        # Add missing keys
        # self.token = None
        # self.register_buffer("old_data_ent_embeddings_weight_frezzed", torch.zeros([[]]))
        # self.register_buffer("old_data_rel_embeddings_weight_frezzed", torch.zeros([[]]))

    def pre_snapshot(self, optimizer, snapshot_id):
        """
        Prepare the model and optimizer before transitioning to the next snapshot.
        
        Args:
            optimizer (torch.optim.Optimizer): The current optimizer instance.
            snapshot_id (int): The current snapshot index.
        
        Returns:
            torch.optim.Optimizer: Reinitialized optimizer for the next snapshot.
        """
        if snapshot_id > 0:
            print(f"Snapshot {snapshot_id}: Resetting for new snapshot...")

            # Step 1: Freeze token if it exists
            if hasattr(self, 'token'):
                self.ent_token.requires_grad = False
                print("Token frozen for new snapshot.")

            # Step 2: Unfreeze embeddings
            for param in self.ent_embeddings.parameters():
                param.requires_grad = True
            for param in self.rel_embeddings.parameters():
                param.requires_grad = True
            print("Embeddings are now trainable.")

            # Step 3: Reinitialize token
            print("Reinitializing token...")
            self.ent_token = nn.Parameter(
                torch.randn(self.args.token_num, self.args.emb_dim, device=self.args.device).normal_(0, 0.01)
            )
            self.ent_token.requires_grad = False

            # Step 4: Reset optimizer with embeddings and token

            optimizer = torch.optim.Adam(
                [param for name, param in self.named_parameters() if "embeddings" in name],
                lr=self.args.learning_rate
            )
            print("Optimizer reset: Training embeddings and reinitialized token.")

        else:
            optimizer = torch.optim.Adam(
                [param for name, param in self.named_parameters() if "embeddings" in name],
                lr=0.1
            )
            print("Snapshot 0: No changes made to optimizer or model parameters.")

        return optimizer


    def init_old_weight(self):
        '''
        Initialize the learned parameters for storage.
        '''
        for name, param in self.named_parameters():
            name_ = name.replace('.', '_')
            if 'embeddings' in name_:
                # self.register_buffer('old_weight_{}'.format(name_), torch.tensor([[]]))
                self.register_buffer('old_data_{}'.format(name_), torch.tensor([[]]))


    def store_old_parameters(self):
        """ store last result """
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            # if param.requires_grad:
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

class TransE(DLKGE):
    def __init__(self, args, kg) -> None:
        super(TransE, self).__init__(args, kg)
        self.huber_loss = torch.nn.HuberLoss(reduction='sum')
        self.ent_token = None


    def add_token(self, optimizer):
        """
        Add self.token to the optimizer while ensuring embeddings are frozen
        and excluded from parameter updates.
        """
        # Add the token parameter to the optimizer
        self.ent_token = nn.Parameter(
            torch.randn(self.args.token_num, self.args.emb_dim, device=self.args.device).normal_(0, 0.01)
        )
        
        # Ensure embeddings are frozen
        for param in self.ent_embeddings.parameters():
            param.requires_grad = False
        for param in self.rel_embeddings.parameters():
            param.requires_grad = False

        # Remove existing parameter groups for embeddings (if any)
        new_param_groups = []
        for param_group in optimizer.param_groups:
            new_params = [p for p in param_group['params'] if p.requires_grad]
            if new_params:
                param_group['params'] = new_params
                new_param_groups.append(param_group)
        
        # Clear the optimizer and re-add valid parameter groups
        optimizer.param_groups = new_param_groups
        
        # Add token to optimizer explicitly
        optimizer.add_param_group({'params': self.ent_token})
        
        print("Token added to optimizer, embeddings excluded successfully.")


    # def freeze_model(self):
    #     """
    #     Always re-register frozen embeddings as buffers to handle dynamic size changes.
    #     """
    #     for name, param in self.named_parameters():
    #         if "token" not in name:
    #             print(f'{name} is frozen and saved in buffer old_data_{name.replace(".", "_")}_frezzed')
                
    #             buffer_name = f'old_data_{name.replace(".", "_")}_frezzed'
                
    #             # Always delete and re-register the buffer
    #             if hasattr(self, buffer_name):
    #                 delattr(self, buffer_name)
                
    #             self.register_buffer(buffer_name, param.data.clone().detach())
    #             # print(f'buffer_name:',buffer_name,self.old_data_ent_embeddings_weight_frezzed.size())
    #             param.requires_grad = False
    #         else:
    #             print(f'{name} is training')
    #             param.requires_grad = True

    # def reset_optimizer(self, optimizer):
    #     """
    #     Reinitialize optimizer for snapshots beyond the first.

    #     Args:
    #         optimizer (torch.optim.Optimizer): Current optimizer instance.
    #         snapshot_id (int): The current snapshot index.
        
    #     Returns:
    #         torch.optim.Optimizer: Reinitialized optimizer for subsequent snapshots.
    #     """

    #     # Freeze token
    #     if hasattr(self, 'token'):
    #         self.token.requires_grad = False

    #     # Unfreeze embeddings
    #     for param in self.ent_embeddings.parameters():
    #         param.requires_grad = True
    #     for param in self.rel_embeddings.parameters():
    #         param.requires_grad = True

    #     # Create a new optimizer for embeddings only
    #     optimizer = torch.optim.Adam(
    #         [param for name, param in self.named_parameters() if "token" not in name],
    #         lr=0.001
    #     )

    #     print("Optimizer reset: Training embeddings only.")
    #     return optimizer


    # def unfreeze_model(self):
    #     for name, param in self.named_parameters():
    #         if "token" not in name:
    #             print(f'{name} is trained')
    #             param.requires_grad = True
    #         else:
    #             print(f'{name} is freezed')
    #             param.requires_grad = False

    def get_TransE_loss(self, head, relation, tail, label,first_layer_mask,second_layer_mask):
        """ Pair wise margin loss: L1-norm (h + r - t) """
        ent_embeddings, rel_embeddings = self.embedding()

        if self.ent_token is not None and self.args.snapshot == 0:
            # Clone and detach embeddings to ensure frozen weights remain unchanged
            frozen_ent_embeddings = ent_embeddings.clone().detach()
            frozen_rel_embeddings = rel_embeddings.clone().detach()
            
            # Attention mechanism
            attention_mask = self.ent_token @ frozen_ent_embeddings.T  # [T, N]
            attention_mask = attention_mask.sigmoid()  # [T, N]
            attention_mask = attention_mask.T.unsqueeze(-1)  # [N, T, 1]
            
            frozen_ent_embeddings = frozen_ent_embeddings.unsqueeze(1)  # [N, 1, D]
            attention_embeddings = frozen_ent_embeddings * attention_mask  # [N, T, D]
            summed_attention_embeddings = torch.sum(attention_embeddings, dim=1)  # [N, D]
            
            # Index selected entities and relations
            h = torch.index_select(summed_attention_embeddings, 0, head)
            r = torch.index_select(frozen_rel_embeddings, 0, relation)
            t = torch.index_select(summed_attention_embeddings, 0, tail)
            
            # Compute score
            score = self.score_fun(h, r, t)
            p_score, n_score = self.split_pn_score(score, label)
            y = torch.Tensor([-1]).to(self.args.device)
            
            return self.margin_loss_func(p_score, n_score, y) / head.size(0)
        if self.args.snapshot > 0:
            if self.ent_token.requires_grad:
                frozen_ent_embeddings = ent_embeddings.clone().detach()
                frozen_rel_embeddings = rel_embeddings.clone().detach()
                
                # Attention mechanism
                attention_mask = self.ent_token @ frozen_ent_embeddings.T  # [T, N]
                attention_mask = attention_mask.sigmoid()  # [T, N]
                attention_mask = attention_mask.T.unsqueeze(-1)  # [N, T, 1]
                
                frozen_ent_embeddings = frozen_ent_embeddings.unsqueeze(1)  # [N, 1, D]
                attention_embeddings = frozen_ent_embeddings * attention_mask  # [N, T, D]
                summed_attention_embeddings = torch.sum(attention_embeddings, dim=1)  # [N, D]
                
                # Index selected entities and relations
                h = torch.index_select(summed_attention_embeddings, 0, head)
                r = torch.index_select(frozen_rel_embeddings, 0, relation)
                t = torch.index_select(summed_attention_embeddings, 0, tail)
                
                # Compute score
                score = self.score_fun(h, r, t)
                p_score, n_score = self.split_pn_score(score, label)
                y = torch.Tensor([-1]).to(self.args.device)
                
                return self.margin_loss_func(p_score, n_score, y) / head.size(0)


        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)/ head.size(0)
        # return self.new_loss(head, relation, tail, label,first_layer_mask,second_layer_mask)

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

    def get_multi_layer_loss(self, entity_mask, relation_mask, entity_mask_weight):
        """ count multy layer loss """
        if self.args.snapshot == 0 or (self.args.use_two_stage and self.args.epoch < self.args.two_stage_epoch_num):
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
        new_ent_embedidngs = self.ent_embeddings.weight * entity_mask.unsqueeze(1)
        # embedding_loss = self.mse_loss_func(new_ent_embedidngs,old_ent_embeddings)
        # print(f'embedding_loss:',embedding_loss)
        loss = self.huber_loss(old_ent_embeddings, new_ent_embedidngs)
        if self.args.using_relation_distill:
            old_rel_embeddings = self.old_data_rel_embeddings_weight * relation_mask.unsqueeze(1)
            new_rel_embeddings = self.rel_embeddings.weight * relation_mask.unsqueeze(1)
            loss += self.huber_loss(old_rel_embeddings, new_rel_embeddings)
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

    def dice_coeff(self,inputs, eps=1e-12):
        """
        Compute Dice Coefficient Loss for token-entity masks.
        
        Args:
            inputs (torch.Tensor): Input mask with shape [T, N]
            eps (float): Small value to prevent division by zero.
        
        Returns:
            torch.Tensor: Dice Coefficient Loss.
        """
        # Step 1: Pairwise comparison between tokens
        pred = inputs[:, None, :]  # [T, 1, N]
        target = inputs[None, :, :]  # [1, T, N]

        # Step 2: Create a mask to exclude self-comparisons
        mask = torch.ones((inputs.size(0), inputs.size(0)), device=inputs.device)
        mask.fill_diagonal_(0)  # Exclude self-comparisons

        # Step 3: Calculate intersection and union for each token pair
        a = torch.sum(pred * target, dim=-1)  # Intersection: [T, T]
        b = torch.sum(pred * pred, dim=-1) + eps  # Sum of squares of pred: [T, T]
        c = torch.sum(target * target, dim=-1) + eps  # Sum of squares of target: [T, T]

        # Step 4: Compute the Dice Coefficient
        d = (2 * a) / (b + c)  # Dice Coefficient: [T, T]

        # Step 5: Apply the mask to exclude self-comparisons
        d = (d * mask).sum() / mask.sum()
        
        # Step 6: Return Dice Loss (1 - Dice Coefficient)
        return 1-d


    def get_token_distilliation_loss(self, entity_mask, relation_mask, entity_mask_weight):
        """ count multy layer loss """
        if self.args.snapshot == 0:
            dice_loss = self.dice_coeff(self.ent_token)
            return 0.0, dice_loss
        if self.args.snapshot > 0:
            if self.ent_token.requires_grad:
                dice_loss = self.dice_coeff(self.ent_token)
                # print('ejfej')
                return 0.0, dice_loss
        # if self.args.use_multi_layers and self.args.using_mask_weight:
        #     new_entity_mask_weight = self.entity_weight_linear(entity_mask_weight[-self.num_new_entity:])
        #     entity_mask[-self.num_new_entity:] = entity_mask[-self.num_new_entity:].clone() * new_entity_mask_weight

        #     entity_mask = torch.ones_like(entity_mask) * self.multi_layer_weight
        old_ent_len = self.kg.snapshots[self.args.snapshot - 1].num_ent
        old_token=self.old_data_token.clone().detach()
        old_ent_embeddings = self.old_data_ent_embeddings_weight[:old_ent_len]*entity_mask[:old_ent_len].unsqueeze(1)
        new_ent_embeddings = self.ent_embeddings.weight[:old_ent_len]*entity_mask[:old_ent_len].unsqueeze(1)
        # print(f'old_ent_embeddings:',old_ent_embeddings.size(),entity_mask[:old_ent_len].size())
        old_mask = old_token @ old_ent_embeddings.T #[T,D]@[N,D]-->[T,N]
        old_mask = old_mask.sigmoid()
        new_mask = old_token @ new_ent_embeddings.T #[T,D]@[N,D]-->[T,N]
        new_mask = new_mask.sigmoid()
        mask = old_mask * new_mask #[T,N]
        # mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)  # Normalize mask along N dimension

        # total_loss = distillation_loss + 0.1 * dice_loss  # Weight Dice Loss (0.1 is adjustable)


        Attentioned_old_ent_embedding = mask.unsqueeze(-1)*old_ent_embeddings.unsqueeze(0) #[T,N,1]@[1,N,D]-->[T,N,D]
        Attentioned_new_ent_embedding = mask.unsqueeze(-1)*new_ent_embeddings.unsqueeze(0) #[T,N,1]@[1,N,D]-->[T,N,D]
        loss = (Attentioned_old_ent_embedding - Attentioned_new_ent_embedding)**2
        # print(f'loss:',loss)
        loss = loss.sum((1))  # [T, D]
        loss = loss / mask.sum((-1)).unsqueeze(-1)
        loss = loss.mean(-1).sum(-1)
        return loss,0

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
        loss = 0.0;decompose_loss = 0.0;MAE_loss = 0.0;multi_layer_loss=0.0
        """ 0. count initial loss """
        if not self.args.using_reply or self.args.snapshot == 0:
            transE_loss = self.get_TransE_loss(head, relation, tail, label,first_layer_mask,second_layer_mask)
            loss += transE_loss
        """ 1. incremental distillation """
        # if self.args.use_multi_layers and (not self.args.without_multi_layers):
        #     multi_layer_loss = self.get_multi_layer_loss(entity_mask, relation_mask, entity_mask_weight)* float(self.args.multi_layer_weight)
        #     loss += multi_layer_loss
        # if self.args.without_multi_layers:
        #     multi_layer_loss = self.get_one_layer_loss() * self.args.embedding_distill_weight
        #     loss += multi_layer_loss
        if self.args.using_token_distillation_loss and self.ent_token is not None:
            decompose_loss,multi_layer_loss = self.get_token_distilliation_loss(entity_mask, relation_mask, entity_mask_weight)
            decompose_loss = decompose_loss * float(self.args.token_distillation_weight)
            loss += decompose_loss
        # if self.args.use_multi_layers and (not self.args.without_multi_layers):
            multi_layer_loss = multi_layer_loss* float(self.args.multi_layer_weight)
            loss += multi_layer_loss
        return transE_loss,multi_layer_loss,MAE_loss,decompose_loss,loss

# if __name__ == "__main__":
#     from src.parse_args import args
#     from src.data_load.KnowledgeGraph import KnowledgeGraph
#     kg = KnowledgeGraph(args)
#     model = TransE(args=args, kg=kg)
#     model.pre_snapshot()