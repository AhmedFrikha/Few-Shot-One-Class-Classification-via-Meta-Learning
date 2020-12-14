# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12

from models.ocsvm_class import OCSVM

from utils import set_gpu, Timer, count_accuracy, check_dir, log

# Code was adapted from "Meta-Learning with Differentiable Convex Optimization"
# Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, Stefano Soatto CVPR 2019


def binary_acc(out, target):
    pred = out >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum().item() / target.numel()
    return acc


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(
                avg_pool=False,
                drop_rate=0.1,
                dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        else:
            network = resnet12(
                avg_pool=False,
                drop_rate=0.1,
                dropblock_size=2).cuda()
    else:
        print("Cannot recognize the network type")
        assert(False)

    # Choose the classification head
    if options.head == 'OC-SVM':
        cls_head = ClassificationHead(base_learner='OC-SVM').cuda()
    elif options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print("Cannot recognize the dataset type")
        assert(False)

    return (network, cls_head)


def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        data_loader = FewShotDataloader
    else:
        print("Cannot recognize the dataset type")
        assert(False)

    return (dataset_train, dataset_val, data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60,
                        help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                        help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                        help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                        help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,
                        help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                        help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                        help='number of classes in one training episode')
    parser.add_argument(
        '--test-way',
        type=int,
        default=5,
        help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument(
        '--network',
        type=str,
        default='ProtoNet',
        help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument(
        '--head',
        type=str,
        default='ProtoNet',
        help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument(
        '--dataset',
        type=str,
        default='miniImageNet',
        help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                        help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                        help='epsilon of label smoothing')

    # OC-SVM specific
    parser.add_argument('--val_oc_svm', type=int, default=1,
                        help='whether to validate using oc-svm')
    parser.add_argument('--val-shot-ocsvm', type=int, default=10,
                        help='number of support examples per validation class')
    parser.add_argument('--val-query-ocsvm', type=int, default=50,
                        help='number of query examples per validation class')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for the experiment')

    opt = parser.parse_args()
    seed = opt.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,
        # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 200,  # num of batches per epoch
        #    epoch_size=1
    )
    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,
        # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    dloader_val_ocsvm = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot_ocsvm,  # num training examples per novel category
        nTestNovel=opt.val_query_ocsvm * opt.test_way,
        # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    # load saved model

#    saved_models = torch.load('./experiments/CB_FC100_resnet/best_model.pth')
#    embedding_net.load_state_dict(saved_models['embedding'])
#    cls_head.load_state_dict(saved_models['head'])

    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 #               {'params': cls_head_cb.parameters()},
                                 {'params': cls_head.parameters()}], lr=0.1, momentum=0.9, \
                                weight_decay=5e-4, nesterov=True)

    def lambda_epoch(e): return 1.0 if e < 20 else (
        0.06 if e < 30 else 0.012 if e < 40 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.num_epoch + 1):
        if(opt.val_oc_svm and (epoch == 1)):
            # Evaluate on the validation split
            _, _ = [x.eval() for x in (embedding_net, cls_head)]

            val_accuracies = []
            val_f1, val_rec, val_prec, val_spec = [], [], [], []
            seed = 2
            oc_svm = OCSVM(seed, linear_only=True, GridSearch=True)
            count_rbf, count_linear = 0, 0
            g_list, nu_list = [], []
            for i, batch in enumerate(tqdm(dloader_val_ocsvm(epoch)), 1):
                data_support, labels_support, data_query, labels_query, _, _ = batch

                # determine normal indices
                normal_indices = np.where(labels_support.numpy() == 0)
                data_support = data_support[normal_indices]
                test_n_support = len(data_support)

                labels_support = np.squeeze(np.array(labels_support.numpy()))
                labels_query = np.squeeze(np.array(labels_query.numpy()))

                if(opt.test_way > 2):
                    # make query set class-balanced (the anomaly class includes
                    # examples from multiple classes from the original dataset)
                    normal_indices_query = np.where(labels_query == 0)[0]
                    anomalous_indices_query = np.where(labels_query != 0)[0]

                    labels_support = np.array(
                        [l if l == 0 else 1 for l in list(labels_support)])
                    labels_query = np.array(
                        [l if l == 0 else 1 for l in list(labels_query)])
                    selected_indices = []
                    selected_indices += list(normal_indices_query)
                    selected_anomalous_indices = random.sample(
                        list(anomalous_indices_query), len(normal_indices_query))
                    selected_indices += list(selected_anomalous_indices)
                    data_query_selected, labels_query_selected = data_query[
                        0][selected_indices], labels_query[selected_indices]

                emb_support = embedding_net(data_support.cuda().reshape(
                    [-1] + list(data_support.shape[-3:])))
                emb_query = embedding_net(data_query_selected.cuda().reshape(
                    [-1] + list(data_query_selected.shape[-3:])))
                acc, prec, rec, spec, f1_score, auc_roc, rbf_better, best_hp = oc_svm.val_op(emb_support.cpu(
                ).detach().numpy(), labels_support, emb_query.cpu().detach().numpy(), labels_query_selected)
                if(rbf_better):
                    count_rbf += 1
                    g_list.append(best_hp)
                else:
                    count_linear += 1
                    nu_list.append(best_hp)

                val_accuracies.append(acc)
                val_f1.append(f1_score)
                val_rec.append(rec)
                val_prec.append(prec)
                val_spec.append(spec)

            val_acc_avg = np.mean(np.array(val_accuracies))
            val_f1_avg = np.mean(np.array(val_f1))
            val_rec_avg = np.mean(np.array(val_rec))
            val_prec_avg = np.mean(np.array(val_prec))
            val_spec_avg = np.mean(np.array(val_spec))
            if(g_list):
                avg_g = np.mean(g_list)
            else:
                avg_g = 100
            if(nu_list):
                avg_nu = np.mean(nu_list)
            else:
                avg_nu = 100

            # we compute the best nu hyperparameter value using the
            # meta-validation tasks
            print(
                'epoch ',
                epoch,
                'val- ocsvm - acc = ',
                val_acc_avg,
                'f1',
                val_f1_avg,
                'rec',
                val_rec_avg,
                'prec',
                val_prec_avg,
                'spec',
                val_spec_avg,
                'rbf',
                count_rbf,
                'linear',
                count_linear,
                'avg_g',
                avg_g,
                'avg_nu',
                avg_nu)

        # Train on the training split
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
            epoch, epoch_learning_rate))

        _, _ = [x.train() for x in (embedding_net, cls_head)]

        train_accuracies = []
        train_losses = []
        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):

            if('OC' in opt.head):
                # the head is an OC-SVM
                data_support, labels_support, data_query, labels_query, _, _ = batch

                labels_query = labels_query.float()
                normal_indices = np.where(labels_support.numpy() == 0)
                data_support = data_support[normal_indices].reshape(
                    [opt.episodes_per_batch, opt.train_shot] + list(data_support.shape[-3:]))

                train_n_support = data_support.shape[1]
                train_n_query = opt.train_way * opt.train_query

                if(opt.train_way > 2):
                    # make query set class-balanced (the anomaly class includes
                    # examples from multiple classes from the original dataset)
                    normal_indices_query = np.where(labels_query.numpy() == 0)
                    anomalous_indices_query = np.where(
                        labels_query.numpy() != 0)
                    data_query_n = data_query[normal_indices_query].reshape(
                        [opt.episodes_per_batch, opt.train_query] + list(data_query.shape[-3:]))
                    data_query_a = data_query[anomalous_indices_query].numpy()
                    data_query_a = data_query_a.reshape([opt.episodes_per_batch, opt.train_query * (
                        opt.train_way - 1)] + list(data_query.shape[-3:]))[:, :opt.train_query]
                    data_query = torch.cat(
                        [data_query_n, torch.Tensor(data_query_a)], dim=1)

                    q_labels_n = torch.zeros(
                        (opt.episodes_per_batch, opt.train_query))
                    q_labels_a = torch.ones(
                        (opt.episodes_per_batch, opt.train_query))

                    labels_query = torch.cat([q_labels_n, q_labels_a], dim=1)
                    train_n_query = labels_query.shape[-1]

                emb_support = embedding_net(data_support.cuda().reshape(
                    [-1] + list(data_support.shape[-3:])))

                emb_support = emb_support.reshape(
                    (opt.episodes_per_batch, train_n_support, emb_support.shape[-1]))

                emb_query = embedding_net(data_query.cuda().reshape(
                    [-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(
                    (opt.episodes_per_batch, train_n_query, -1))

                labels_query = labels_query.cuda()
                logit_query = cls_head(
                    emb_query, emb_support, None, None, opt.train_shot)

                logit_query = logit_query.reshape(
                    (opt.episodes_per_batch * train_n_query,)) * (-1.0)
                labels_query = labels_query.reshape(
                    (opt.episodes_per_batch * train_n_query,))

                loss = torch.nn.BCEWithLogitsLoss()(
                    logit_query, labels_query.type_as(logit_query))

                acc = binary_acc(torch.sigmoid(
                    logit_query).reshape(-1), labels_query.reshape(-1))

                train_accuracies.append(acc)
                train_losses.append(loss.item())
                if (i % 100 == 0):
                    train_acc_avg = np.mean(np.array(train_accuracies))
                    train_loss_avg = np.mean(train_losses)
                    log(log_file_path,
                        'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(epoch,
                                                                                                              i,
                                                                                                              len(dloader_train),
                                                                                                              train_loss_avg,
                                                                                                              train_acc_avg,
                                                                                                              acc))
                    if(cls_head.enable_scale):
                        log(log_file_path,
                            'scale: {:.2f}'.format(cls_head.scale.item()))

            else:
                # use original class-balanced meta-training
                data_support, labels_support, data_query, labels_query, _, _ = [
                    x.cuda() for x in batch]

                train_n_support = opt.train_way * opt.train_shot
                train_n_query = opt.train_way * opt.train_query

                emb_support = embedding_net(data_support.reshape(
                    [-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(
                    opt.episodes_per_batch, train_n_support, -1)

                emb_query = embedding_net(data_query.reshape(
                    [-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(
                    opt.episodes_per_batch, train_n_query, -1)

                logit_query = cls_head(
                    emb_query,
                    emb_support,
                    labels_support,
                    opt.train_way,
                    opt.train_shot)

                smoothed_one_hot = one_hot(
                    labels_query.reshape(-1), opt.train_way)
                smoothed_one_hot = smoothed_one_hot * \
                    (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

                log_prb = F.log_softmax(
                    logit_query.reshape(-1, opt.train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()

                acc = count_accuracy(
                    logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

                train_accuracies.append(acc.item())
                train_losses.append(loss.item())

                if (i % 100 == 0):
                    train_acc_avg = np.mean(np.array(train_accuracies))
                    log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} % \t scale: {:.2f})'.format(
                        epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc, cls_head.scale.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        if(not('OC' in opt.head)):
            _, _ = [x.eval() for x in (embedding_net, cls_head)]

            val_accuracies = []
            val_losses = []

            for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
                data_support, labels_support, data_query, labels_query, _, _ = [
                    x.cuda() for x in batch]

                test_n_support = opt.test_way * opt.val_shot
                test_n_query = opt.test_way * opt.val_query

                emb_support = embedding_net(data_support.reshape(
                    [-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)
                emb_query = embedding_net(data_query.reshape(
                    [-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)

                logit_query = cls_head(
                    emb_query,
                    emb_support,
                    labels_support,
                    opt.test_way,
                    opt.val_shot)

                loss = x_entropy(
                    logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
                acc = count_accuracy(
                    logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

                val_accuracies.append(acc.item())
                val_losses.append(loss.item())

            val_acc_avg = np.mean(np.array(val_accuracies))
            val_acc_ci95 = 1.96 * \
                np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

            val_loss_avg = np.mean(np.array(val_losses))
            if val_acc_avg > max_val_acc:
                max_val_acc = val_acc_avg
                torch.save({'embedding': embedding_net.state_dict(
                ), 'head': cls_head.state_dict()}, os.path.join(opt.save_path, 'best_model.pth'))
                log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'
                    .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
            else:
                log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'
                    .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

            torch.save({'embedding': embedding_net.state_dict(),
                        'head': cls_head.state_dict()},
                       os.path.join(opt.save_path,
                                    'last_epoch.pth'))
            if epoch % opt.save_epoch == 0:
                torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict(
                )}, os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

            log(log_file_path,
                'Elapsed Time: {}/{}\n'.format(timer.measure(),
                                               timer.measure(epoch / float(opt.num_epoch))))

        # Evaluate on the validation split
        if(opt.val_oc_svm):
            _, _ = [x.eval() for x in (embedding_net, cls_head)]

            val_accuracies = []
            val_f1, val_rec, val_prec, val_spec = [], [], [], []
            seed = 2
            oc_svm = OCSVM(seed, linear_only=True, GridSearch=True)
            count_rbf, count_linear = 0, 0
            g_list, nu_list = [], []
            for i, batch in enumerate(tqdm(dloader_val_ocsvm(epoch)), 1):
                data_support, labels_support, data_query, labels_query, _, _ = batch

                # determine normal indices
                normal_indices = np.where(labels_support.numpy() == 0)
                data_support = data_support[normal_indices]
                test_n_support = len(data_support)

                labels_support = np.squeeze(np.array(labels_support.numpy()))
                labels_query = np.squeeze(np.array(labels_query.numpy()))

                if(opt.test_way > 2):
                    # make query set class-balanced (the anomaly class includes
                    # examples from multiple classes from the original dataset)

                    normal_indices_query = np.where(labels_query == 0)[0]
                    anomalous_indices_query = np.where(labels_query != 0)[0]
                    labels_support = np.array(
                        [l if l == 0 else 1 for l in list(labels_support)])
                    labels_query = np.array(
                        [l if l == 0 else 1 for l in list(labels_query)])
                    selected_indices = []
                    selected_indices += list(normal_indices_query)
                    selected_anomalous_indices = random.sample(
                        list(anomalous_indices_query), len(normal_indices_query))
                    selected_indices += list(selected_anomalous_indices)
                    data_query_selected, labels_query_selected = data_query[
                        0][selected_indices], labels_query[selected_indices]

                emb_support = embedding_net(data_support.cuda().reshape(
                    [-1] + list(data_support.shape[-3:])))
                emb_query = embedding_net(data_query_selected.cuda().reshape(
                    [-1] + list(data_query_selected.shape[-3:])))
                acc, prec, rec, spec, f1_score, auc_roc, rbf_better, best_hp = oc_svm.val_op(emb_support.cpu(
                ).detach().numpy(), labels_support, emb_query.cpu().detach().numpy(), labels_query_selected)
                if(rbf_better):
                    count_rbf += 1
                    g_list.append(best_hp)
                else:
                    count_linear += 1
                    nu_list.append(best_hp)

                val_accuracies.append(acc)
                val_f1.append(f1_score)
                val_rec.append(rec)
                val_prec.append(prec)
                val_spec.append(spec)

            val_acc_avg = np.mean(np.array(val_accuracies))
            val_f1_avg = np.mean(np.array(val_f1))
            val_rec_avg = np.mean(np.array(val_rec))
            val_prec_avg = np.mean(np.array(val_prec))
            val_spec_avg = np.mean(np.array(val_spec))
            if(g_list):
                avg_g = np.mean(g_list)
            else:
                avg_g = 100
            if(nu_list):
                avg_nu = np.mean(nu_list)
            else:
                avg_nu = 100

            print(
                'epoch ',
                epoch,
                'val- ocsvm - acc = ',
                val_acc_avg,
                'f1',
                val_f1_avg,
                'rec',
                val_rec_avg,
                'prec',
                val_prec_avg,
                'spec',
                val_spec_avg,
                'rbf',
                count_rbf,
                'linear',
                count_linear,
                'avg_g',
                avg_g,
                'avg_nu',
                avg_nu)

            if val_acc_avg > max_val_acc:
                print('model saved')
                max_val_acc = val_acc_avg
                torch.save({'embedding': embedding_net.state_dict(
                ), 'head': cls_head.state_dict()}, os.path.join(opt.save_path, 'best_model.pth'))
                log(log_file_path, 'Validation Epoch: {}\t\t\tAccuracy: {:.2f} % (Best)'
                    .format(epoch, val_acc_avg))
            else:
                log(log_file_path, 'Validation Epoch: {}\t\t\tAccuracy: {:.2f} %'
                    .format(epoch, val_acc_avg))

            torch.save({'embedding': embedding_net.state_dict(),
                        'head': cls_head.state_dict()},
                       os.path.join(opt.save_path,
                                    'last_epoch.pth'))
            if epoch % opt.save_epoch == 0:
                torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict(
                )}, os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

            log(log_file_path,
                'Elapsed Time: {}/{}\n'.format(timer.measure(),
                                               timer.measure(epoch / float(opt.num_epoch))))
