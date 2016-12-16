import pandas as pd
import goldsberry
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor
import sklearn.linear_model as sl
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as m
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import itertools
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

scores = ['accuracy', 'average_precision', 'fl', 'f1_macro', 'neg_log_loss', 'precision', 'recall', 'roc_auc']

Seasons = ['2012-13', '2013-14', '2014-15', '2015-16']
Game_Data = {}
gameids = goldsberry.GameIDs()
for i in range(len(Seasons)):
    gameids.get_new_data(Season=Seasons[i])
    Game_Data[Seasons[i]] = pd.DataFrame(gameids.game_list())

team_ids = list(Game_Data[Seasons[0]]['TEAM_ID'].value_counts().index)
WL_dict = {'W': 1, 'L': -1}


def stats_per_season(season_index):
    print "Creating Stats For Season %s" % (Seasons[season_index])
    goldsberry.apiparams.p_team_season['Season'] = Seasons[season_index]
    data = [goldsberry.team.season_stats(team).overall()[0] for team in team_ids]
    season_data = pd.DataFrame(data)
    season_data = season_data.set_index(['TEAM_ID'])
    return season_data


#
# Stats_Data = {}
#
# for i in range(len(Seasons)):
#     Stats_Data[Seasons[i]] = stats_per_season(i)
#




def find_clusters(features):
    kmeans = KMeans(n_clusters=5).fit(features)
    return kmeans


def visualize_clusters(features):
    range_n_clusters = [2, 3, 4, 5, 6]
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(features) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(features)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(features, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(features, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(features[features.columns[0]], features[features.columns[1]], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()


def stats_per_season_quarter(season_index, quarter):
    print "Creating Stats Per Season Per Quarter for season %s quarter %s" % (Seasons[season_index], quarter)
    goldsberry.apiparams.p_team_season['Season'] = Seasons[season_index]
    goldsberry.apiparams.p_team_season['Period'] = quarter
    data = [goldsberry.team.season_stats(team).overall()[0] for team in team_ids]
    quarter_data = pd.DataFrame(data)
    quarter_data = quarter_data.set_index(['TEAM_ID'])
    goldsberry.apiparams.p_team_season['Period'] = 0
    return quarter_data


#
# Stats_Data_Quarter = [{}, {}, {}]
# for i in range(1,4):
#     for j in range(len(Seasons)):
#         Stats_Data_Quarter[i][Seasons[j]] = stats_per_season_quarter(j, i)
#


# get there stats in the first quarter of the game
# get there win streak,

# def stats_per_interval(season_index, SeasonSegment='Pre All-Star', start_date='', end_date='',Period=0):
def stats_per_interval(season_index, start_date='', end_date='', Period=0):
    # set params
    print "Creating Stats Per Time Interval for season %s" % (season_index)
    goldsberry.apiparams.p_team_season['Season'] = Seasons[season_index]
    goldsberry.apiparams.p_team_season['Period'] = Period
    goldsberry.apiparams.p_team_season['DateFrom'] = start_date
    goldsberry.apiparams.p_team_season['DateTo'] = end_date

    data = [goldsberry.team.season_stats(team).overall()[0] for team in team_ids]
    per_interval = pd.DataFrame(data).set_index(['TEAM_ID'])

    # reset params
    goldsberry.apiparams.p_team_season['DateTo'] = ''
    goldsberry.apiparams.p_team_season['DateFrom'] = ''
    goldsberry.apiparams.p_team_season['Period'] = 0
    return per_interval


def in_season_X(season_index, percentage_of_season):
    df = Game_Data[Seasons[season_index]]
    df['GAME_DATE'] = df['GAME_DATE'].apply(lambda x: pd.to_datetime(x))
    df = df.sort_values('GAME_DATE').set_index('GAME_DATE')

    dates = [(x.strftime('%Y-%m-%d')) for x in list(df.index.value_counts().index.sort_values())]

    start_date = dates[0]
    mid_index = int(len(dates) / 2)

    end_date = dates[mid_index]
    print start_date,
    print end_date
    Data = stats_per_interval(season_index, start_date, end_date)

    X = pd.DataFrame()
    Y = []
    print "Creating IN Season Data for Season %s" % (season_index)
    unique_game_ids = list(df.ix[end_date:]['GAME_ID'].value_counts().index)
    for i, game in enumerate(unique_game_ids):
        print "doing game %s out of games %s in season %s" % (i, len(unique_game_ids), season_index)
        print game
        try:
            rows = df[['WL', 'TEAM_ID']].ix[df['GAME_ID'] == game]
            stat_columns = [x for x in list(Data.columns) if
                            'TEAM_NAME' not in x and 'GROUP_SET' not in x and 'GROUP_VALUE' not in x and 'GP' not in x]
            tmp = goldsberry.game.boxscore_summary(game).game_summary()[0]
            home_team_id = tmp['HOME_TEAM_ID']
            visitor_team_id = tmp['VISITOR_TEAM_ID']
            home_team_data = Data.ix[tmp['HOME_TEAM_ID']][stat_columns]
            away_team_data = Data.ix[tmp['VISITOR_TEAM_ID']][stat_columns]

            away_team_data = pd.DataFrame([list(away_team_data.values)], index=[game],
                                          columns=[x + '_AWAY' for x in list(away_team_data.index)])
            home_team_data = pd.DataFrame([list(home_team_data.values)], index=[game],
                                          columns=[x + '_HOME' for x in list(home_team_data.index)])
            data = pd.concat([home_team_data, away_team_data], 1)
            last_matchup = goldsberry.game.boxscore_summary(game).last_meeting()[0]
            data['DIFFERENTIAL_LAST_MATCHUP'] = last_matchup['LAST_GAME_HOME_TEAM_POINTS'] - last_matchup[
                'LAST_GAME_VISITOR_TEAM_POINTS']
            X = X.append(data)

            if rows['WL'].ix[rows['TEAM_ID'] == tmp['HOME_TEAM_ID']].values[0] == 'W':
                Y.append(1)
            else:
                Y.append(0)
        except (TypeError, ValueError):
            print 'issue with game id %s' % game
            pass

    return X, Y


def create_X(season_index, Data=None, quarter=0):
    df = Game_Data[Seasons[season_index]]
    unique_game_ids = list(df['GAME_ID'].value_counts().index)

    if Data:
        if season_index != 0:
            prev_season = season_index - 1
            avg_season_data = Data[Seasons[prev_season]]
        else:
            print("We Need Prev Data For Stats")
            return
    else:
        avg_season_data = stats_per_season_quarter(season_index, quarter)

    X = pd.DataFrame()
    Y = []
    print "Creating X for Season %s quarter %s" % (season_index, quarter)
    for i, game in enumerate(unique_game_ids):
        print "doing game %s out of games %s in season %s" % (i, len(unique_game_ids), season_index)
        rows = df[['WL', 'TEAM_ID']].ix[df['GAME_ID'] == game]
        stat_columns = [x for x in list(avg_season_data.columns) if
                        'TEAM_NAME' not in x and 'GROUP_SET' not in x and 'GROUP_VALUE' not in x and 'GP' not in x]
        tmp = goldsberry.game.boxscore_summary(game).game_summary()[0]
        home_team_data = avg_season_data.ix[tmp['HOME_TEAM_ID']][stat_columns]
        away_team_data = avg_season_data.ix[tmp['VISITOR_TEAM_ID']][stat_columns]
        away_team_data = pd.DataFrame([list(away_team_data.values)], index=[game],
                                      columns=[x + '_AWAY' for x in list(away_team_data.index)])
        home_team_data = pd.DataFrame([list(home_team_data.values)], index=[game],
                                      columns=[x + '_HOME' for x in list(home_team_data.index)])
        data = pd.concat([home_team_data, away_team_data], 1)
        X = X.append(data)
        if rows['WL'].ix[rows['TEAM_ID'] == tmp['HOME_TEAM_ID']].values[0] == 'W':
            Y.append(1)
        else:
            Y.append(0)

    return X, Y


def normalize_data(X):
    X_norm = (X - X.mean()) / (X.max() - X.min())

    return X_norm


def predict_it(X_train, y_train, X_test, Y_test):
    ESTIMATORS = {
        "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                           random_state=0),
        "K-nn": KNeighborsRegressor(),
        "Linear regression": LinearRegression(),
        "Ridge": RidgeCV(),
    }
    y_test_predict = dict()
    for name, estimator in ESTIMATORS.items():
        estimator.fit(X_train, y_train)
        y_test_predict[name] = estimator.predict(X_test)


def TreeRegressor(X, y):
    df_norm = (X - X.mean()) / (X.max() - X.min())
    forest = ExtraTreesRegressor()
    clf = forest.fit(X, y)
    clf.feature_importances_
    X_new = clf.transform(X)

    # model = SelectFromModel(clf, prefit=True)
    Yhat = clf.predict(X)


def game_X(season_index, start_range, end_range):
    g_data = Game_Data[Seasons[season_index]]
    unique_game_ids = list(g_data['GAME_ID'].value_counts().index)

    goldsberry.apiparams.p_game_bs['RangeType'] = 2
    goldsberry.apiparams.p_game_bs['EndRange'] = end_range
    goldsberry.apiparams.p_game_bs['StartRange'] = start_range

    X = pd.DataFrame()
    Y = []
    print "Creating quarter Season Data for Season %s ranges %s \t %s" % (season_index, start_range, end_range)
    for j, game_id in enumerate(unique_game_ids):
        print "doing game %s out of games %s in season %s startrange %s" % (
        j, len(unique_game_ids), season_index, start_range)
        rows = g_data[['WL', 'TEAM_ID']].ix[g_data['GAME_ID'] == game_id]

        tmp = goldsberry.game.boxscore_summary(game_id).game_summary()[0]

        home_team_id = tmp['HOME_TEAM_ID']
        visitor_team_id = tmp['VISITOR_TEAM_ID']

        df = pd.DataFrame(goldsberry.game.boxscore_advanced(game_id).team_stats())
        df = df.set_index('TEAM_ID')
        df = df.drop(['TEAM_ABBREVIATION', 'TEAM_NAME', 'TEAM_CITY', 'MIN', 'GAME_ID'], 1)
        home_team_data = df.ix[home_team_id]
        away_team_data = df.ix[visitor_team_id]
        away_team_data = pd.DataFrame([list(away_team_data.values)], index=[game_id],
                                      columns=[x + '_AWAY' for x in list(away_team_data.index)])
        home_team_data = pd.DataFrame([list(home_team_data.values)], index=[game_id],
                                      columns=[x + '_HOME' for x in list(home_team_data.index)])
        data = pd.concat([home_team_data, away_team_data], 1)
        X = X.append(data)
        if rows['WL'].ix[rows['TEAM_ID'] == tmp['HOME_TEAM_ID']].values[0] == 'W':
            Y.append(1)
        else:
            Y.append(0)

    return X, Y


def normalize_data(X):
    X_norm = (X - X.mean()) / (X.max() - X.min())
    return X_norm


def train_predict(estimator, X_train, y_train, X_test, y_test):
    if estimator.__str__().split("(")[0] in ['SVC', 'NuSVC']:
        y_train_svm = y_train.copy()
        y_test_svm = y_test.copy()
        y_train_svm[y_train==0] = -1
        y_test_svm[y_test==0] = -1
        yhat = estimator.fit(X_train, y_train_svm).predict(X_test)
        return (y_test_svm, yhat)
    else:
        yhat =estimator.fit(X_train, y_train).predict(X_test)


    return (y_test,yhat)

def train_predict_crossval(estimator, X_train, y_train):

    yhat = cross_val_predict(estimator, X_train, y_train, cv=10)

    return yhat

def feature_selection(X_train, y_train):

    ESTIMATORS = {
        "Extra Trees": ExtraTreesClassifier(n_estimators=10, max_features=32, random_state=0),
        # "Random Forest": RandomForestClassifier()
    }

    for name, estimator in ESTIMATORS.iteritems():
        clf = estimator.fit(X_train, y_train)
        model = SelectFromModel(clf, prefit=True)


    indices = model.get_support(True)
    model_fit = clf.score(X_train, y_train, sample_weight=None)

    return indices, model_fit


def plot_confusion_matrix(cm, classes,classifier,
                          normalize=False,
                          title='Confusion_matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title = title + '_' + classifier
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('Images/' + title + '.png', bbox_inches='tight')
    plt.close()



scores = ['accuracy', 'average_precision', 'fl', 'f1_macro', 'neg_log_loss', 'precision', 'recall', 'roc_auc']


def predict_it_crossval(X, Y, data_name):
    ESTIMATORS = {
        "Extra trees": ExtraTreesClassifier(n_estimators=10, random_state=0),
        "K-nn with 10 neighbors and 30 leafs": KNeighborsClassifier(n_neighbors=10),
        # "Ridge": RidgeCV(alphas=(0.1, 1.0, 3.0, 5.0, 7.0, 10.0)),
        "SGD Classifier": sl.SGDClassifier(loss='modified_huber', penalty='elasticnet', l1_ratio=0.15, n_iter=5,
                                           shuffle=True, verbose=False, n_jobs=10, average=False,
                                           class_weight='balanced'),
        "Logistic Regression": sl.LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X_train.shape[0]),
        "SVM with rbf kernel": svm.SVC(kernel='rbf', C=2.5,probability=True),
        "SVM with linear kernel": svm.SVC(kernel='linear', C=2.5,probability=True),
        # "SVM with polynomial kernel": svm.SVC(kernel='poly'),
        "SVM with sigmoid": svm.SVC(kernel='sigmoid', C=2.5,probability=True),
        "Non-linear SVM": svm.NuSVC(probability=True),
        "Neural Net": MLPClassifier(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(5, 2), random_state=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,
                                                        random_state=0)

    }
    # m.auc,
    metrics = [m.matthews_corrcoef, m.f1_score,
               m.recall_score, m.zero_one_loss, m.precision_score, m.recall_score, m.mean_squared_error,
               m.accuracy_score, m.f1_score]
    scores = ['accuracy', 'average_precision', 'f1', 'f1_macro', 'neg_log_loss', 'precision', 'recall', 'roc_auc']
    results_r = {}
    results_p = {}
    indices = {}
    model_fit = {}

    df_r_c = []
    df_p_c = []
    confusion_r = {}
    confusion_p = {}
    for name, estimator in ESTIMATORS.iteritems():
        print name
        indices[name], model_fit[name] = feature_selection(X, Y)
        df_p_c.append(
            {scores[x]: cross_val_score(estimator, X[indices[name]], Y, cv=10, scoring=scores[x]).mean() for x in range(len(scores))})
        df_r_c.append(
            {scores[x]: cross_val_score(estimator, X, Y, cv=10, scoring=scores[x]).mean() for x in range(len(scores))})



        results_r[name] = (Y, train_predict_crossval(estimator, X, Y))
        results_p[name] = (Y,train_predict_crossval(estimator, X[indices[name]], Y))
        plot_learning_curve(estimator, name + '_LearningCurve_' + data_name + '_reg', X, Y)
        plot_learning_curve(estimator, name + '_LearningCurve_' + data_name + '_feature_selection', X[indices[name]], Y)
        do_roc(estimator,X,Y, name + '_ROC_CURVE' + data_name + '_reg' )
        do_roc(estimator, X[indices[name]], Y, name + '_ROC_CURVE_' + data_name + '_feature_selection')

    df_r = []
    df_p = []
    confusion_r = {}
    confusion_p = {}
    for (name_r, result_r), (name_p, result_p) in zip(results_r.items(), results_p.items()):
        df_r.append({metrics[i].__name__: metrics[i](result_r[0], result_r[1]) for i in range(len(metrics))})
        df_p.append({metrics[i].__name__: metrics[i](result_p[0], result_p[1]) for i in range(len(metrics))})

        confusion_r[name_r] = m.confusion_matrix(result_r[0], result_r[1])
        confusion_p[name_r + '_Feature_Selection'] = m.confusion_matrix(result_p[0], result_p[1])

    df_p = pd.DataFrame(df_p, index=results_p.keys())
    df_r = pd.DataFrame(df_r, index=results_r.keys())
    df_p_c = pd.DataFrame(df_p_c, index=results_p.keys())
    df_r_c = pd.DataFrame(df_r_c, index=results_r.keys())


    # confusion_p = pd.DataFrame(confusion_p, index = results_p.keys())
    # confusion_r = pd.DataFrame(confusion_r, index=results_r.keys())

    return results_r, results_p, df_p, df_r, df_p_c,df_r_c, confusion_p, confusion_r




def predict_it(X_train, y_train, X_test, Y_test):
    ESTIMATORS = {
        "Extra trees": ExtraTreesClassifier(n_estimators=10, random_state=0),
        "K-nn with 10 neighbors and 30 leafs": KNeighborsClassifier(n_neighbors=10),
        # "Ridge": RidgeCV(alphas=(0.1, 1.0, 3.0, 5.0, 7.0, 10.0)),
        "SGD Classifier": sl.SGDClassifier(loss='hinge', penalty='elasticnet',l1_ratio=0.15, n_iter=5, shuffle=True, verbose=False, n_jobs=10, average=False, class_weight='balanced'),
        "Logistic Regression": sl.LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X_train.shape[0]),
        "SVM with rbf kernel": svm.SVC(kernel='rbf', C = 2.5),
        "SVM with linear kernel": svm.SVC(kernel='linear',C = 2.5),
        # "SVM with polynomial kernel": svm.SVC(kernel='poly'),
        "SVM with sigmoid": svm.SVC(kernel='sigmoid',C = 2.5),
        "Non-linear SVM": svm.NuSVC(),
        "Neural Net": MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)


    }
    # m.auc,
    metrics = [m.matthews_corrcoef, m.f1_score,
               m.recall_score, m.zero_one_loss, m.precision_score, m.recall_score, m.mean_squared_error,m.accuracy_score, m.f1_score]
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    results_r = {}
    results_p = {}
    indices = {}
    model_fit = {}
    for name, estimator in ESTIMATORS.iteritems():
        indices[name], model_fit[name] = feature_selection(X_train, y_train)
        results_p[name] = train_predict(estimator, X_train[indices[name]], y_train, X_test[indices[name]], Y_test)
        results_r[name] = train_predict(estimator, X_train, y_train, X_test, Y_test)


    df_r = []
    df_p = []
    confusion_r = {}
    confusion_p = {}
    for (name_r, result_r),(name_p, result_p) in zip(results_r.items(), results_p.items()):
        df_r.append({metrics[i].__name__: metrics[i](result_r[0], result_r[1]) for i in range(len(metrics))})
        df_p.append({metrics[i].__name__: metrics[i](result_p[0], result_p[1]) for i in range(len(metrics))})

        confusion_r[name_r] = m.confusion_matrix(result_r[0], result_r[1])
        confusion_p[name_r + '_Feature_Selection'] =  m.confusion_matrix(result_p[0], result_p[1])



    df_p = pd.DataFrame(df_p, index = results_p.keys())
    df_r = pd.DataFrame(df_r, index= results_r.keys())
    # confusion_p = pd.DataFrame(confusion_p, index = results_p.keys())
    # confusion_r = pd.DataFrame(confusion_r, index=results_r.keys())

    return results_r, results_p, df_p, df_r, confusion_p,confusion_r




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=10,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('ImagesLearningCurve/' + title + '.png', bbox_inches='tight')
    plt.close()



def do_roc(estimator,X,Y,title):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.5,
                                                        random_state=0)
    print estimator.__str__().split("(")[0]
    if estimator.__str__().split("(")[0] in ['SVC', 'NuSVC']:
        y_score = estimator.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = estimator.fit(X_train, y_train).predict_proba(X_test)
        y_score = y_score[:,1]


    plot_roc(estimator, 2, y_test, y_score, title)




def plot_roc(classifier, n_classes, Y_test, Y_score,title):


    fpr, tpr,_ = roc_curve(Y_test, Y_score,pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('ImagesROC/' + title + '.png', bbox_inches='tight')
    plt.close()



def main():
    # # for i in range(1,len(Seasons)):
    # #     X, y = create_X(i,Stats_Data)
    # #     X['Y_VALUES'] = y
    # #     X.to_pickle(Seasons[i] + '_FULL_GAME.pickle')
    # #
    # #
    # # for i in range(1,len(Seasons)):
    # #     for j in range(1,4):
    # #         X, y = create_X(i,None, quarter=j)
    # #         X['Y_VALUES'] = y
    # #         X.to_pickle(Seasons[i] + '_QUARTER_' + str(j) + '_.pickle')
    # #
    #
    # # season_percentage = .5
    # # for i in range(1,len(Seasons)):
    # #         X, y = in_season_X(i,season_percentage)
    # #         X['Y_VALUES'] = y
    # #         X.to_pickle(Seasons[i] + 'HALF_SEASON' +'_.pickle')

    # ranges = [(7200*x,7200*y) for x,y in zip(range(1,4),range(2,5))]
    # halfs =  [(1400*x,1400*y) for x,y in zip(range(0,2),range(1,3))]
    # for i in range(2, len(Seasons)):
    #     for r in ranges:
    #         X,y = game_X(i, r[0], r[1])
    #         X.to_pickle(Seasons[i] + '_' + str(r[0]) + '_' + str(r[1]) + '_.pickle')
    #
    pwd = os.getcwd()
    reg_data = [x for x in os.listdir(pwd) if '_QUARTER_2_.pickle' in x or '_QUARTER_3_.pickle' in x or 'HALF_SEASON_.pickle' in x]
    data_big = [pd.read_pickle(lol) for lol in reg_data]


    for i,data in enumerate(data_big):

        print " ----------------------------------"
        # new = pd.read_pickle('2014-15_QUARTER_2_.pickle')
        # new =  new.rename(columns = {x:'q2' + x for x in list(new.columns) })
        # new2 = pd.read_pickle('2013-14_FULL_GAME.pickle')
        # new2 = new2.rename(columns = {x:'q3' + x for x in list(new2.columns) })
        # reg = pd.read_pickle(data_set)
        #       # train = list(data.index[np.random.randint(0, data.shape[0], size=int(data.shape[0]*.8))])
        # # test = []
        # # for x in data.index:
        # data = pd.concat([reg, new, new2],axis=1)


        #     if x not in train:
        #         test.append(x)
        x_cols = [col for col in data.columns if
                  col not in ['Y_VALUES', 'MIN_HOME', 'MIN_AWAY', 'q2Y_VALUES', 'q2MIN_HOME', 'q2MIN_AWAY','q3Y_VALUES', 'q3MIN_HOME', 'q3MIN_AWAY']]
        data_X = normalize_data(data[x_cols])

        y_col = 'Y_VALUES'
        data_Y = data[y_col]
        # kf = KFold(n_splits=2)

        #results_r, results_p, df_p, df_r, confusion_p, confusion_r = predict_it(X_train,Y_Train,X_test,Y_Test)

        results_r, results_p, df_p, df_r, df_p_c, df_r_c, confusion_p, confusion_r = predict_it_crossval(data_X, data_Y,reg_data[i])
        file_name = 'results/Results_' + reg_data[i].split('.pickle')[0] + '_CrossVal'
        df_p_c.to_csv(file_name + '_feature_selection.csv')
        df_r_c.to_csv(file_name + '_regular.csv')
        file_name = 'results2/Results_' + reg_data[i].split('.pickle')[0] + '_CrossVal'
        df_p.to_csv(file_name + '_feature_selection.csv')
        df_r.to_csv(file_name + '_regular.csv')
        for (name_r, cm_r),(name_p, cm_p) in zip(confusion_r.items(), confusion_p.items()):
            plot_confusion_matrix(cm_r, ['Win', 'Loss'], name_r + reg_data[i].split('.pickle')[0] ,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues)

            plot_confusion_matrix(cm_p, ['Win', 'Loss'], name_p + reg_data[i].split('.pickle')[0] ,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)


# #        #df_p.sort_values(by=list(df_p.columns), ascending=False)                                                                                                   X_test,
    #                                                                                                           Y_Test)
    #     print {i:mean_error[i] for i in sorted(mean_error,key= lambda x:mean_error[x]) }
    #     print {i:mean_error_selec_feat[i] for i in sorted(mean_error_selec_feat,key= lambda x:mean_error_selec_feat[x]) }
    #     print indices


main()
