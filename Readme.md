Due to the amount of requirements of code  with respect to **GraphTransfer**, we therefore release the key implementation, i.e., cross fusion module, so as to further facilitate the development of research.

We hope this code helps you well. If you use this code in your work, please cite our paper.

```
GraphTransfer: A Generic Feature Fusion Framework for Collaborative Filtering
Jiafeng Xia, Dongsheng Li, Hansu Gu, Tun Lu and Ning Gu.
arXiv:2408.05792. 2024
```



### The implementation of Cross Fusion Model

The Cross Fusion Module is used to distill knowledge from a pretrained model (which we called the meta model), so as to benefit the learning of current model (which we called the main model) in a "learn to fuse" manner.

The whole training process can refer to Algorithm 1 in our paper, which is split into two cascaded stages: the training of the meta model, and the training of the main model. 

Suppose we have obtained user and item embeddings, denoted as `metamodel_user_emb` and `metamodel_item_emb` from the meta model, as well as user and item embeddings, denoted as `h_user` and `h_item` from the main model, then we can use the following code to fuse different features  in a "learn-to-fuse" manner

```python
# Preliminaries:
# 	metamodel_user_emb: [M, D]
# 	metamodel_item_emb: [N, D]
# 	h_user: [B, D]
# 	h_item: [B, D]
# 	user_item_pair: [B, 2], containing user historical interactions in one batch, the first column is user id, the second column is item id.
# 	train_loss_1: the loss of main model
# 	mainmodel_optimizer: the optimizer of main model

# Step 1: Obtain predicted scores from the meta model.
metamodel_scores = torch.sum(torch.mul(metamodel_user_emb[user_item_pair[0]], metamodel_item_emb[user_item_pair[1]]), dim=1, keepdim=False)

# Step 2: Obtain transfer scores and the corresponding losses between the meta model and the main model through the cross-dot-product mechanism
transfer_scores_1 = torch.sum(torch.mul(metamodel_user_emb[user_item_pair[0]], h_item[user_item_pair[1]]), dim=1, keepdim=False)
train_loss_2 = torch.mean(torch.square(transfer_scores_1 - metamodel_scores))

transfer_scores_2 = torch.sum(torch.mul(h_user[user_item_pair[0]], metamodel_item_emb[user_item_pair[1]]), dim=1, keepdim=False)
train_loss_3 = torch.mean(torch.square(transfer_scores_2 - metamodel_scores))

# Step 3: Train the model
train_loss = train_loss_1 + lambda_1 * train_loss_2+ lambda_2 * train_loss_3

mainmodel_optimizer.zero_grad()
train_loss.backward()
mainmodel_optimizer.step()
```

