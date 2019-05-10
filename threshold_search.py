## search threshold
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from score import f2_score

def threshold_search(labels, preds, do_plot=False):
    score = []
    thrs = np.arange(0, 0.5, 0.05)
    for thr in tqdm(thrs):
        score.append(f2_score(labels, preds, thr))
    score = np.array(score)
    pm = score.argmax()
    best_thr, best_score = thrs[pm], score[pm].item()
    print('thr={best_thr:%.3f}' % best_thr, 'F2={best_score:%.3f}' % best_score)
    if do_plot:
        plt.plot(thrs, score)
        plt.vlines(x=best_thr, ymin=score.min(), ymax=score.max())
        plt.text(best_thr+0.03, best_score-0.01, f'$F_{2}=${best_score:.3f}', fontsize=14);
        plt.show()
    return best_thr, best_score
