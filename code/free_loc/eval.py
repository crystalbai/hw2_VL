import sklearn.metrics


def compute_map(gt, pred, average=None):
    """
    Compute the multi-label classification accuracy.
    gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
        image.
    pred (np.ndarray): Shape Nx20, probability of that object in the image
        (output probablitiy).
    """
    nclasses = gt.shape[1]
    all_ap = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid]
        pred_cls = pred[:, cid]
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        all_ap.append(ap)
    return all_ap
def compute_curve(gt, pred, average=None):
    """
    Compute the multi-label classification accuracy.
    gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
        image.
    pred (np.ndarray): Shape Nx20, probability of that object in the image
        (output probablitiy).
    """
    nclasses = gt.shape[1]
    all_ap = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid]
        pred_cls = pred[:, cid]
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        f1 = sklearn.metrics.f1_score(gt_cls, pred_cls)
        all_ap.append(f1)
    return all_ap