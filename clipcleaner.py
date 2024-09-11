import os.path

import clip
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from datasets import get_specific_dataset


# two different sample selection mechanisms
def get_score(prediction, labels, mode='celoss'):
    '''

    :param prediction: we can take the prediction from LR or KNN, ot utilize the CLIP text probing similarity directly
    :param labels:
    :param mode: 'celoss', 'consistency'
    :return:
    '''
    num_classes = len(np.unique(labels))
    if mode == 'celoss':
        loss = prediction.log()
        score = torch.gather(loss, 1, labels.view(-1, 1)).squeeze()
    elif mode == 'perclass_celoss':
        loss = prediction.log()
        score = torch.gather(loss, 1, labels.view(-1, 1)).squeeze()
        id_by_label = [np.where(labels == i)[0] for i in range(num_classes)]
        for id in id_by_label:
            score[id] = (score[id] - score[id].min()) / (score[id].max() - score[id].min())
    else:  # if mode =='consistency':
        vote_y = torch.gather(prediction, 1, labels.view(-1, 1)).squeeze()
        vote_max = prediction.max(dim=1)[0]
        score = vote_y / vote_max
    return score


def logistic_regression(features, labels):
    # by default, we use weighted logistic regression to counter for the class imbalance
    classifier = LogisticRegression(random_state=0, max_iter=10000, class_weight='balanced').fit(features.cpu(), labels)  # .cpu())
    prediction = torch.tensor(classifier.predict_proba(features.cpu()))
    return prediction


def combined_selection(args):
    '''

    :param args:
    :return: selected_id, nonselected_id
    '''
    # clip_model, preprocess = clip.load("ViT-L/14@336px")
    # print('calculate new labels!')
    if args.model == 'small':
        clip_model, preprocess = clip.load("ViT-B/32")
    elif args.model == 'tiny':
        clip_model, preprocess = clip.load("RN50")
    else:
        clip_model, preprocess = clip.load("ViT-L/14@336px")
    clip_data = get_specific_dataset(args, preprocess, mode='train', noise_mode=args.noise_mode, noise_ratio=args.noise_ratio, type='red')
    num_classes = clip_data.num_classes
    clip_loader = torch.utils.data.DataLoader(clip_data, batch_size=1024, shuffle=False, num_workers=8)

    detailed_features = clip_data.detailed_features
    class_names = clip_data.class_names
    suffix = clip_data.suffix
    all_features = []
    all_labels = torch.tensor(clip_data.label)
    with torch.no_grad():
        for i, [data, label, index] in enumerate(clip_loader):
            # print(i)
            image_features = clip_model.encode_image(data.cuda()).float()
            all_features.append(image_features)
            # all_labels.append(label)
            # if i % 100 == 1:
            #     # break
            #     print('data loading step: ', i)

    all_features = torch.cat(all_features, dim=0).detach()
    all_features = F.normalize(all_features, dim=1) # N x d
    del clip_loader

    # %% Option 1: zero-shot classification
    text_tokens = [
        [clip.tokenize(f"a photo of a {class_names[i][0].lower()} {j}" + suffix).cuda() if type(class_names[0]) == list else clip.tokenize(f"a photo of a {class_names[i].lower()} {j}" + suffix).cuda() for j in
         detailed_features[i]] for i in range(num_classes)]
    text_features = [[clip_model.encode_text(text_token_i).float().detach() for text_token_i in text_token] for text_token in text_tokens] # K x Num of prompts x d
    similarity = torch.zeros(len(all_features), num_classes) # N x K
    with torch.no_grad():
        for i in range(num_classes):
            text_features_i = torch.cat(text_features[i], dim=0) # concatenate multiple prompts: Num of prompts x d
            text_features_i = F.normalize(text_features_i, dim=1)
            similarity_i1 = torch.einsum('ac, bc->ab', all_features, text_features_i) # sample to prompt similarity: N x d , Num of prompts x d -> N x Num of prompts
            similarity[:, i] = torch.exp(similarity_i1 / 0.07).sum(1)  # taking summation over different prompts for class i: N x Num of prompts -> N x 1
    prediction_zero = (similarity / similarity.sum(1, keepdim=True)).cpu()

    # %% Option 2: logistic regression with only visual features
    prediction_lr = logistic_regression(all_features.cpu(), all_labels)

    # extract sample selection scores
    CLIP_similarityprob_loss = get_score(prediction_zero, all_labels, 'perclass_celoss')
    CLIP_similarityprob_consistency = get_score(prediction_zero, all_labels, 'consistency')
    CLIP_visuallr_loss = get_score(prediction_lr, all_labels, 'perclass_celoss')
    CLIP_visuallr_consistency = get_score(prediction_lr, all_labels, 'consistency')

    # %% save scores
    method_score = [
        CLIP_similarityprob_loss, CLIP_similarityprob_consistency,
        CLIP_visuallr_loss, CLIP_visuallr_consistency]

    method_name = [
        'zeroshot_perclassgmm', 'zeroshot_consistency',
        'visuallr_perclassgmm', 'visuallr_consistency']

    # take intersection by default
    types = ['loss', 'consistency',
             'loss', 'consistency']

    id_by_label = [np.where(all_labels == i)[0] for i in range(num_classes)]

    select_i = []
    # loss-based scores ---> GMM
    all_labels = all_labels.numpy()
    for i, score in enumerate(method_score):
        clean_id_all = []
        for k in range(num_classes):  # per-class sample selection
            if types[i] == 'loss':
                gmm = GaussianMixture(2)
                gmm.fit(score[id_by_label[k]].reshape(-1, 1))
                prob = gmm.predict_proba(score[id_by_label[k]].reshape(-1, 1))[:, gmm.means_.argmax()]
                clean_id = id_by_label[k][np.where(prob >= args.theta_gmm)[0]]
            else:
                clean_id = id_by_label[k][np.where(score[id_by_label[k]] >= args.theta_cons)[0]]
            clean_id_all.append(clean_id)
        clean_id_all = np.concatenate(clean_id_all)
        per_class = np.array([np.sum(all_labels[clean_id_all] == i) for i in range(num_classes)])
        print('Per-class sample selection results: ', per_class)

        select_i.append(clean_id_all)

    # aggregate final sample selection results
    select = select_i[0]
    for i in range(len(select_i) - 1):
        select = np.intersect1d(select, select_i[i + 1])

    per_class = np.array([np.sum(all_labels[select] == i) for i in range(num_classes)])
    print(method_name[i], per_class.min(), per_class.mean(), per_class.max(), per_class.sum())

    ################################### To avoid empty class, fill in with clip zeroshot classifier selection #######################################
    num_by_class = np.array([np.sum(all_labels[select] == i) for i in range(num_classes)])
    print(num_by_class)
    # at least K samples should be clean! To avoid too small selection for specific class!
    num_smallest = int(len(all_labels) / num_classes / num_classes)
    zero_class = np.where(num_by_class == 0)[0]
    if len(zero_class) != 0:
        non_zero_class = np.where(num_by_class != 0)[0]
        num_smallest = num_by_class[non_zero_class].min()
        if num_smallest < len(all_labels) / num_classes / num_classes:
            num_smallest = int(len(all_labels) / num_classes / num_classes)
        # print(num_smallest)
        # print(zero_class, non_zero_class, num_smallest)
        all_by_class = [np.where(all_labels == i)[0] for i in range(num_classes)]
        for clx in zero_class:
            clx_prob = method_score[0][all_by_class[clx]]
            # prob_rank = clx_prob.argsort(descending=True)
            prob_rank = clx_prob.argsort()
            if num_smallest == 1:
                selected_clx = np.array(all_by_class[clx][prob_rank[:num_smallest]])
            else:
                selected_clx = all_by_class[clx][prob_rank[:num_smallest]]
            select = np.concatenate([select, selected_clx])
    select = np.unique(select)
    per_class = np.array([np.sum(all_labels[select] == i) for i in range(num_classes)])
    print(method_name[i], per_class.min(), per_class.mean(), per_class.max(), per_class.sum())

    return select, all_labels[select] # selected samples index, selected samples labels
