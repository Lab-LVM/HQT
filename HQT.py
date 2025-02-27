import os.path
import random
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from models.utils import load_backbone
from data.utils import get_dataset


class RandomTree:
    def __init__(self, maxDepth=5, split_fn='K_Means', minDataNum=2, seed=0):
        assert maxDepth > 0
        assert maxDepth < 30
        assert minDataNum > 1

        self.maxDepth = maxDepth
        self.split_fn = split_fn
        self.minDataNum = minDataNum
        self.seed = seed
        self.node = {}

    def build_node(self, features, index, depth):
        if depth >= self.maxDepth or features.shape[0] < self.minDataNum:
            self.node.update({str(index) + '_node': 'leaf'})
            self.node.update({str(index) + '_node_index': index})
            return

        if self.split_fn == 'K_Means':
            kmeans = KMeans(n_clusters=2, n_init='auto', random_state=self.seed).fit(features)
            left_index = kmeans.labels_ == 0
            right_index = kmeans.labels_ == 1
            self.node.update({str(index) + '_kmeans': kmeans})

        self.node.update({str(index) + '_node': 'normal'})

        left_features = features[left_index, :]
        right_features = features[right_index, :]

        self.build_node(left_features, index * 2 + 1, depth + 1)
        self.build_node(right_features, index * 2 + 2, depth + 1)

    def build_tree(self, features):
        depth = 0
        self.build_node(features, 0, depth)

    def predict(self, feature, index=0):
        assert index >= 0
        node = self.node[str(index)+'_node']

        if node=="normal":
            if self.split_fn == 'K_Means':
                kmeans = self.node[str(index) + '_kmeans']
                child = kmeans.predict(feature)
            return self.predict(feature, index * 2 + int(child) + 1)

        elif node=="leaf":
            return self.node[str(index) + '_node_index']

def getfeature(args):
    dataset = get_dataset(args.dataset_type, root=args.data_path,
                             num_query=args.num_query,
                             num_train=args.num_train,
                             batch_size=args.batch_size,
                             num_workers=args.workers,
                             hash_model=args.hash_model,
                             mean=args.mean,
                             std=args.std,
                             img_size=args.img_size,
                             scale=args.scale,
                             get_feature=True)
    if 'vgg' in args.arch:
        if os.path.exists(args.feature_path):
            features = np.load(args.feature_path, allow_pickle=True)
        else:
            model = load_backbone(args.arch, use_timm=args.use_timm).to(args.device)
            features = np.empty(shape=[len(dataset.dataset.targets), 4096])
            cursor = 0
            model.eval()
            for batch_step, (img, img_aug, lbs, index) in enumerate(tqdm(dataset, desc='Feature extracted.')):
                img = img.to(args.device)
                if args.use_timm:
                    x = model.forward_features(img)
                    x = model.pre_logits(x)
                    x = x.squeeze()
                else:
                    x = model.features(img)
                    x = x.view(x.size(0), -1)
                    x = model.classifier(x)
                features[cursor:cursor+x.shape[0], :] = x.cpu().detach().numpy()
                cursor = cursor + x.shape[0]
            np.save(args.feature_path, features)
        return features
    else:
        raise ValueError("If you want another model.")

def HQT(features, code_path, encode_length, min_depth=3, max_depth=5, seed=0):
    if os.path.exists(code_path):
        raw_lbs = np.load(code_path)
    else:
        raw_lbs = np.empty(shape=[len(features), encode_length])
        for i in tqdm(range(encode_length), desc="Building Tree."):
            tree_depth = random.randrange(min_depth, max_depth+1)
            rt = RandomTree(maxDepth=tree_depth, seed=seed)
            rt.build_tree(features)
            for f in range(features.shape[0]):
                lbs = rt.predict(features[f, :].reshape(1, features.shape[1]))
                raw_lbs[f, i] = lbs
        np.save(code_path, raw_lbs)
    labels = np.empty(shape=[len(features), encode_length])
    for i in range(encode_length):
        rlbs = raw_lbs[:, i]
        urlbs = np.unique(rlbs)
        urlbs = np.random.permutation(urlbs)
        rlbs[np.isin(rlbs, urlbs[:len(urlbs) // 2])] = -1
        rlbs[np.isin(rlbs, urlbs[len(urlbs) // 2:])] = +1
        labels[:, i] = rlbs
    return labels