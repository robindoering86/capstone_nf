def plot_corr_matrix(corr, cmap = None, mask_upper = True, show_annot = False, figsize = (12, 12), title = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    if mask_upper:
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = None
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    # Generate a custom diverging colormap
    if cmap is None:
        cmap = sns.diverging_palette(240, 10, n=5, sep=50, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        #vmax=1,
        robust=True,
        center=0,
        square=True,
        linewidths=.2,
        cbar_kws={'shrink': .6},
        annot = show_annot,
    )
    ax.set_ylim(corr.shape[0], 0)
    title = 'Heatmap of Correlation between Features' if title is None else title
    plt.title(title)
    f.tight_layout()
    



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import plotly.graph_objects as go
    
    
    fn = cm[1][0]
    fp = cm[0][1]
    tn = cm[0][0]
    tp = cm[1][1]

    accuracy = (tn+tp)/(tn+tp+fn+fp)
    misclass = 1 - accuracy
    
    # positive predictive value (ppv) / precision
    ppv = tp/(tp+fp)
    fdr = fp/(tp+fp)
    
    # npv
    npv = tn/(tn+fn)
    tdr = fn/(tn+fn)
    
    # tpr / recall
    tpr = tp/(tp+fn)
    fnr = 1 - tpr
    # tnr / specificity^
    tnr = tn/(tn+fp)
    fpr = 1 - tnr

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
    #layout = go.Layout()
    fig = go.Figure(
        data=go.Heatmap(
            z=cm
        )
    )
    fig.show()
    
def plot_roc_auc_curve(fpr, tpr):
    """
    """
    plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
    plt.plot([0,1],[0,1],'k-',label='random')
    plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    
def plot_feature_importances(importances, columns, model, fname=None, sort=True, top_count=10):
    """
    """
    import matplotlib.pyplot as plt

    if sort:
        importances, columns = zip(*sorted(zip(importances, columns)))
    plt.figure(figsize=(8, min(int(len(importances[-top_count:])/2), 20)))
    plt.barh(range(len(importances[-top_count:])), importances[-top_count:], align='center') 
    plt.yticks(range(len(columns[-top_count:])), columns[-top_count:]) 
    plt.title(f'Feature importances in the {model} model')
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    if fname:
        plt.savefig(fname)
    plt.show()

def plot_multi(df, ncols=2, title=None, l0=0, l1=1):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    nlevelzero = len(df.columns.get_level_values(l0).unique())
    nlevelone = len(df.columns.get_level_values(l1).unique().drop('Volume'))

    nrows = nlevelzero//ncols + nlevelzero%ncols
    
    level0_cols = df.columns.get_level_values(l0).unique()
    level1_cols = df.columns.get_level_values(l1).unique().drop('Volume')
    plt.figure(figsize=(10, 40))
    for i, level0 in enumerate(level0_cols):
        plt.subplot(nrows, ncols, i+1)
        for i, level1 in enumerate(level1_cols):
            sns.lineplot(x=df.index, y=level1, markers=True, dashes=False, data=df[level0])
        plt.title(level0)
    plt.tight_layout()
    plt.title(title)
    plt.show()
        

def plot_norm_dist(data, cnt):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import norm
    
    mean, std = norm.fit(data)
    #plt.hist(x, bins=30, normed=True,)
    sns.distplot(data, bins=40, kde = False, norm_hist=True)
    xmin, xmax = plt.xlim()
    xx = np.linspace(xmin, xmax, 100)
    y = norm.pdf(xx, mean, std)
    plt.plot(xx, y)
    plt.title(f'Days since: {cnt}\nFit results: Std {std: .4f}; Mean {mean: .4f}')
    plt.show()
