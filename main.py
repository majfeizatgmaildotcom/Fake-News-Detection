import sys
sys.path.append('../')
from LVAE.lvae import *
from LVAE.lda import *
from plots import *
from classifiers import *
from LVAE.preprocessing import *


def vae_experiment(run_info, top_folder):

    main_folder = run_info['main_folder']

    target_column = run_info['target_column']
    dataset_name = run_info['dataset_name']

    word2vec_dim = run_info['word2vec_dim']
    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    x_train = pd.read_csv(this_data_folder + 'x_train_' + str(word2vec_dim) + '.csv')
    x_test = pd.read_csv(this_data_folder + 'x_test_' + str(word2vec_dim) + '.csv')

    if not os.path.exists(main_folder + 'model_weights.hdf5') or \
            not os.path.exists(main_folder + 'model_history.pickle'):
        train_vae(run_info, top_folder)

    # lvae feature extraction run_info, top_folder, filtered_data_df, data_name, col_label
    if not os.path.exists(main_folder+'features/'):
        os.makedirs(main_folder+'features/')

    if not os.path.exists(main_folder + 'features/vae_train.npy') or \
            not os.path.exists(main_folder + 'features/train_label.npy'):

        vae_features_tr, output_tr = extract_features(run_info, top_folder, x_train, data_name='train',
                                                      col_label=target_column, output_label='label')

        np.save(main_folder + 'features/vae_train', vae_features_tr)
        np.save(main_folder + 'train_label', output_tr)

    if not os.path.exists(main_folder + 'features/vae_test.npy') or \
            not os.path.exists(main_folder + 'features/test_label.npy'):

        vae_features_te, output_te = extract_features(run_info, top_folder, x_test, data_name='test',
                                                      col_label=target_column, output_label='label')
        np.save(main_folder + 'features/vae_test', vae_features_te)
        np.save(main_folder + 'test_label', output_te)

    # y, y_pred = test_vae(main_folder, sequence_length, latent_dim, reg_lambda, fnd_lambda, phase='train')


def lda_experiment(run_info, top_folder):
    main_folder = run_info['main_folder']
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    n_topics = run_info['n_topics']
    n_top_words = run_info['n_top_words']
    n_features = run_info['n_features']
    n_iter = run_info['n_iter']
    target_column = run_info['target_column']
    dataset_name = run_info['dataset_name']
    word2vec_dim = run_info['word2vec_dim']
    print('LDA experiment with', n_topics, 'topics:')
    with open(main_folder + 'run_info.pickle', 'wb') as handle:
        pkl.dump(run_info, handle)

    this_data_folder = top_folder + 'data/' + dataset_name + '/'

    x_train = pd.read_csv(this_data_folder + 'x_train_' + str(word2vec_dim) + '.csv')
    x_test = pd.read_csv(this_data_folder + 'x_test_' + str(word2vec_dim) + '.csv')

    # train lda
    data_df_all = x_train.append(x_test, ignore_index=True)

    lda_tf, tf_vectorizer = LDA_train_tf(main_folder, n_topics, data_df_all[target_column], n_top_words,
                                         n_features, n_iter=n_iter)

    # lda_tfidf, tfidf_vectorizer = LDA_train_tf_idf(main_folder, n_topics, data_df_all[target_column],
    #                                                n_top_words, n_features, n_iter=n_iter)

    # LDA
    # tf features
    if not os.path.exists(main_folder+'features/'):
        os.makedirs(main_folder+'features/')
    lda_features_tf_tr = extract_lda_tf_features(main_folder, x_train[target_column], data_name='train')
    lda_features_tf_te = extract_lda_tf_features(main_folder, x_test[target_column], data_name='test')
    np.save(main_folder + 'features/lda_tf_train', lda_features_tf_tr)
    np.save(main_folder + 'features/lda_tf_test', lda_features_tf_te)

    # # tfidf features
    # lda_features_tfidf_tr = extract_lda_tfidf_features(main_folder, X_train[target_column], data_name='train')
    # lda_features_tfidf_te = extract_lda_tfidf_features(main_folder, X_test[target_column], data_name='test')
    # np.save(main_folder + 'features/lda_tfidf_train', lda_features_tfidf_tr)
    # np.save(main_folder + 'features/lda_tfidf_test', lda_features_tfidf_te)


def concat_features(run_info):

    main_folder = run_info['main_folder']
    included_features = run_info['included_features']
    print('Concatenating feature sets in: ', included_features)

    # included_features = ['vae', 'lda_tf', 'lda_tfif']
    address_tr = main_folder + 'features/' + included_features[0] + '_train.npy'
    features_tr = np.load(address_tr)

    address_te = main_folder + 'features/' + included_features[0] + '_test.npy'
    features_te = np.load(address_te)
    feat_name = included_features[0]
    if len(included_features) > 1:

        for feat in included_features[1:]:
            address_tr = main_folder + 'features/' + feat + '_train.npy'
            feat_tr = np.load(address_tr)
            features_tr = np.concatenate((features_tr, feat_tr), axis=1)

            address_te = main_folder + 'features/' + feat + '_test.npy'
            feat_te = np.load(address_te)
            features_te = np.concatenate((features_te, feat_te), axis=1)
            feat_name = feat_name + '_' + feat
    # vae_features_tr = np.load(main_folder + 'features/vae_train.npy')
    # vae_features_te = np.load(main_folder + 'features/vae_test.npy')
    # lda_features_tf_tr = np.load(main_folder + 'features/lda_tf_train.npy')
    # lda_features_tf_te = np.load(main_folder + 'features/lda_tf_test.npy')
    # lda_features_tfidf_tr = np.load(main_folder + 'features/lda_tfidf_train.npy')
    # lda_features_tfidf_te = np.load(main_folder + 'features/lda_tfidf_test.npy')
    #
    # lvae_features_tr = np.concatenate((vae_features_tr, lda_features_tf_tr), axis=1)
    # lvae_features_te = np.concatenate((vae_features_te, lda_features_tf_te), axis=1)

    # save the features
    np.save(main_folder + 'features/lvae_train_' + feat_name, features_tr)
    np.save(main_folder + 'features/lvae_test_' + feat_name, features_te)


def make_results_df(top_folder, results_path, dataset_name):
    # top_folder = 'runs/'
    # dataset_name = 'ISOT' # 'test'
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname]
            # and 'topics_32' in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')[:, 1]
        y_test = np.load(run + '/test_label.npy')[:, 1]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
        classifiers = ['SVM_linear', 'SVM_nonlinear', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                       'KNN (K = 3)', 'linear disriminat Analysis']
        classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                             mlp_classifier, knn3, linear_discriminat_analysis]
        features = ['VAE', 'LDA', 'VAE + LDA']
        datasets = ['Train', 'Test']

        data_tr = [vae_features_train, lda_features_train, data3_train]
        data_te = [vae_features_test, lda_features_test, data3_test]
        len_df = len(classifiers) * len(metrics) * len(datasets) * len(data_tr)
        results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric'], index=range(len_df))
        ct = 0
        for c in range(len(classifiers)):
            classifier_name = classifiers[c]
            classifier = classifiers_funcs[c]
            print(classifier_name)
            for d in range(len(data_tr)):
                this_tr = data_tr[d]
                this_te = data_te[d]
                this_feature = features[d]
                print(this_feature)
                tr, te = classifier(this_tr, this_te, y_train, y_test)
                accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy'
                results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision'
                results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall'
                results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score'
                results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR'
                results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR'
                results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy'
                results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision'
                results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall'
                results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score'
                results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR'
                results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR'
                ct += 12
            results_pd.to_csv(results_path + model_name + '.csv', index=False)


def make_classifiers_df(run_info):
    print('Evaluating the features by classifiers ...')
    main_folder = run_info['main_folder']

    features_path = main_folder + 'features/'
    vae_features_train = np.load(features_path + 'vae_train.npy')
    vae_features_test = np.load(features_path + 'vae_test.npy')
    lda_features_train = np.load(features_path + 'lda_tf_train.npy')
    lda_features_test = np.load(features_path + 'lda_tf_test.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
    y_train = np.load(main_folder + 'train_label.npy')[:, 1]
    y_test = np.load(main_folder + 'test_label.npy')[:, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
    classifiers = ['SVM_linear', 'SVM_nonlinear', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                   'KNN (K = 3)', 'linear disriminat Analysis']
    classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                         mlp_classifier, knn3, linear_discriminat_analysis]
    features = ['VAE', 'LDA', 'VAE + LDA']
    datasets = ['Train', 'Test']

    data_tr = [vae_features_train, lda_features_train, data3_train]
    data_te = [vae_features_test, lda_features_test, data3_test]
    len_df = len(classifiers) * len(metrics) * len(datasets) * len(data_tr)
    results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric'], index=range(len_df))
    ct = 0
    for c in range(len(classifiers)):
        classifier_name = classifiers[c]
        classifier = classifiers_funcs[c]
        # print(classifier_name)
        for d in range(len(data_tr)):
            this_tr = data_tr[d]
            this_te = data_te[d]
            this_feature = features[d]
            # print(this_feature)
            tr, te = classifier(this_tr, this_te, y_train, y_test)
            accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
            accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
            results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy'
            results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision'
            results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall'
            results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score'
            results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR'
            results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR'
            results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy'
            results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision'
            results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall'
            results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score'
            results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR'
            results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR'
            ct += 12
        # results_pd.to_csv(results_path + model_name + '.csv', index=False)
        results_pd.to_csv(main_folder + 'Classifiers.csv', index=False)


def make_classifiers_with_fs_chi2_df(run_info):
    print('Evaluating the features by classifiers and chi2 feature selection...')
    main_folder = run_info['main_folder']
    latent_dim = run_info['latent_dim']
    n_topics = run_info['n_topics']
    if min(latent_dim, n_topics) <= 32:
        check_point_nf = [1]
        [check_point_nf.append(cc) for cc in list(np.arange(0, min(latent_dim, n_topics), 5))[1:]]
        check_point_nf.append(min(latent_dim, n_topics))
    else:
        check_point_nf = [1]
        [check_point_nf.append(cc) for cc in list(np.arange(0, min(latent_dim, n_topics), 10))[1:]]
        check_point_nf.append(min(latent_dim, n_topics))

    features_path = main_folder + 'features/'
    vae_features_train = np.load(features_path + 'vae_train.npy')
    vae_features_test = np.load(features_path + 'vae_test.npy')
    lda_features_train = np.load(features_path + 'lda_tf_train.npy')
    lda_features_test = np.load(features_path + 'lda_tf_test.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
    y_train = np.load(main_folder + 'train_label.npy')[:, 1]
    y_test = np.load(main_folder + 'test_label.npy')[:, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
    classifiers = ['SVM (linear)', 'SVM (nonlinear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                   'KNN (K = 3)', 'linear disriminat Analysis']
    classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                         mlp_classifier, knn3, linear_discriminat_analysis]
    features = ['VAE', 'LDA', 'VAE + LDA']
    datasets = ['Train', 'Test']
    len_df = len(classifiers) * len(metrics) * len(datasets) * len(features) * len(check_point_nf)
    results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric',
                                               '# Selected Features'], index=range(len_df))
    ct = 0
    for nf in check_point_nf:
        # print('Number of features = ', nf)
        new_vae_features_train = SelectKBestFS(vae_features_train, y_train, nf)
        new_vae_features_test = SelectKBestFS(vae_features_test, y_test, nf)
        new_lda_features_train = SelectKBestFS(lda_features_train, y_train, nf)
        new_lda_features_test = SelectKBestFS(lda_features_test, y_test, nf)
        new_data3_train = SelectKBestFS(data3_train, y_train, nf)
        new_data3_test = SelectKBestFS(data3_test, y_test, nf)

        data_tr = [new_vae_features_train, new_lda_features_train, new_data3_train]
        data_te = [new_vae_features_test, new_lda_features_test, new_data3_test]

        for c in range(len(classifiers)):
            classifier_name = classifiers[c]
            classifier = classifiers_funcs[c]
            # print(classifier_name)
            for d in range(len(data_tr)):
                this_tr = data_tr[d]
                this_te = data_te[d]
                this_feature = features[d]
                # print(this_feature)
                tr, te = classifier(this_tr, this_te, y_train, y_test)
                accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy', nf
                results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision', nf
                results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall', nf
                results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score', nf
                results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR', nf
                results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR', nf
                results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy', nf
                results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision', nf
                results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall', nf
                results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score', nf
                results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR', nf
                results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR', nf
                ct += 12
            results_pd.to_csv(main_folder + 'Classifiers_with_chi2_fs.csv', index=False)


def make_classifiers_with_fs_gini_df(run_info):
    print('Evaluating the features by classifiers and gini feature selection ...')
    main_folder = run_info['main_folder']
    latent_dim = run_info['latent_dim']
    n_topics = run_info['n_topics']
    if min(latent_dim, n_topics) <= 32:
        check_point_nf = [1]
        [check_point_nf.append(cc) for cc in list(np.arange(0, min(latent_dim, n_topics), 5))[1:]]
        check_point_nf.append(min(latent_dim, n_topics))
    else:
        check_point_nf = [1]
        [check_point_nf.append(cc) for cc in list(np.arange(0, min(latent_dim, n_topics), 10))[1:]]
        check_point_nf.append(min(latent_dim, n_topics))

    features_path = main_folder + 'features/'
    vae_features_train = np.load(features_path + 'vae_train.npy')
    vae_features_test = np.load(features_path + 'vae_test.npy')
    lda_features_train = np.load(features_path + 'lda_tf_train.npy')
    lda_features_test = np.load(features_path + 'lda_tf_test.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
    y_train = np.load(main_folder + 'train_label.npy')[:, 1]
    y_test = np.load(main_folder + 'test_label.npy')[:, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
    classifiers = ['SVM (linear)', 'SVM (nonlinear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                   'KNN (K = 3)', 'linear disriminat Analysis']
    classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                         mlp_classifier, knn3, linear_discriminat_analysis]
    features = ['VAE', 'LDA', 'VAE + LDA']
    datasets = ['Train', 'Test']
    len_df = len(classifiers) * len(metrics) * len(datasets) * len(features) * len(check_point_nf)
    results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric',
                                               '# Selected Features'], index=range(len_df))
    ct = 0
    for nf in check_point_nf:
        # print('Number of features = ', nf)
        new_vae_features_train = random_forest_feasture_selection(vae_features_train, y_train, nf)
        new_vae_features_test = random_forest_feasture_selection(vae_features_test, y_test, nf)
        new_lda_features_train = random_forest_feasture_selection(lda_features_train, y_train, nf)
        new_lda_features_test = random_forest_feasture_selection(lda_features_test, y_test, nf)
        new_data3_train = random_forest_feasture_selection(data3_train, y_train, nf)
        new_data3_test = random_forest_feasture_selection(data3_test, y_test, nf)

        data_tr = [new_vae_features_train, new_lda_features_train, new_data3_train]
        data_te = [new_vae_features_test, new_lda_features_test, new_data3_test]

        for c in range(len(classifiers)):
            classifier_name = classifiers[c]
            classifier = classifiers_funcs[c]
            # print(classifier_name)
            for d in range(len(data_tr)):
                this_tr = data_tr[d]
                this_te = data_te[d]
                this_feature = features[d]
                # print(this_feature)
                tr, te = classifier(this_tr, this_te, y_train, y_test)
                accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy', nf
                results_pd.loc[ct + 1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision', nf
                results_pd.loc[ct + 2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall', nf
                results_pd.loc[ct + 3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score', nf
                results_pd.loc[ct + 4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR', nf
                results_pd.loc[ct + 5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR', nf
                results_pd.loc[ct + 6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy', nf
                results_pd.loc[ct + 7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision', nf
                results_pd.loc[ct + 8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall', nf
                results_pd.loc[ct + 9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score', nf
                results_pd.loc[ct + 10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR', nf
                results_pd.loc[ct + 11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR', nf
                ct += 12
            results_pd.to_csv(main_folder + 'Classifiers_with_gini_fs.csv', index=False)


def make_results_with_fs_df(top_folder, results_path, dataset_name):
    # top_folder = 'runs/'
    # dataset_name = 'ISOT' # 'test'
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname
            and 'topics_32' in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')[:, 1]
        y_test = np.load(run + '/test_label.npy')[:, 1]
        nfall = lda_features_test.shape[1]
        # nfvae = vae_features_test.shape[1]
        check_point_nf = [1, 5, 10, 15, 20, 25, 30, nfall]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
        classifiers = ['SVM (linear)', 'SVM (nonlinear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                       'KNN (K = 3)', 'linear disriminat Analysis']
        classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                             mlp_classifier, knn3, linear_discriminat_analysis]
        features = ['VAE', 'LDA', 'VAE + LDA']
        datasets = ['Train', 'Test']
        len_df = len(classifiers) * len(metrics) * len(datasets) * len(features) * len(check_point_nf)
        results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric',
                                                   '# Selected Features'], index=range(len_df))
        ct = 0
        for nf in check_point_nf:
            print('Number of features = ', nf)
            new_vae_features_train = SelectKBestFS(vae_features_train, y_train, nf)
            new_vae_features_test = SelectKBestFS(vae_features_test, y_test, nf)
            new_lda_features_train = SelectKBestFS(lda_features_train, y_train, nf)
            new_lda_features_test = SelectKBestFS(lda_features_test, y_test, nf)
            new_data3_train = SelectKBestFS(data3_train, y_train, nf)
            new_data3_test = SelectKBestFS(data3_test, y_test, nf)

            data_tr = [new_vae_features_train, new_lda_features_train, new_data3_train]
            data_te = [new_vae_features_test, new_lda_features_test, new_data3_test]

            for c in range(len(classifiers)):
                classifier_name = classifiers[c]
                classifier = classifiers_funcs[c]
                print(classifier_name)
                for d in range(len(data_tr)):
                    this_tr = data_tr[d]
                    this_te = data_te[d]
                    this_feature = features[d]
                    print(this_feature)
                    tr, te = classifier(this_tr, this_te, y_train, y_test)
                    accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                    accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                    results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy', nf
                    results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision', nf
                    results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall', nf
                    results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score', nf
                    results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR', nf
                    results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR', nf
                    results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy', nf
                    results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision', nf
                    results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall', nf
                    results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score', nf
                    results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR', nf
                    results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR', nf
                    ct += 12
                results_pd.to_csv(results_path + model_name + '_with_fs.csv', index=False)


def make_results_with_randome_forest_fs_df(top_folder, results_path, dataset_name):
    # top_folder = 'runs/'
    # dataset_name = 'ISOT' # 'test'
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname
             and 'topics_32' in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')[:, 1]
        y_test = np.load(run + '/test_label.npy')[:, 1]
        nfall = lda_features_test.shape[1]
        # nfvae = vae_features_test.shape[1]
        check_point_nf = [1, 5, 10, 15, 20, 25, 30, nfall]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F-score', 'FPR', 'FNR']
        classifiers = ['SVM (linear)', 'SVM (nonlinear)', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP',
                       'KNN (K = 3)', 'linear disriminat Analysis']
        classifiers_funcs = [svm_linear, svm_nonlinear, logistic_regression, random_forest_classifier, naive_bayes,
                             mlp_classifier, knn3, linear_discriminat_analysis]
        features = ['VAE', 'LDA', 'VAE + LDA']
        datasets = ['Train', 'Test']
        len_df = len(classifiers) * len(metrics) * len(datasets) * len(features) * len(check_point_nf)
        results_pd = pd.DataFrame(data=0, columns=['Value', 'Classifier', 'Feature', 'Dataset', 'Metric',
                                                   '# Selected Features'], index=range(len_df))
        ct = 0
        for nf in check_point_nf:
            print('Number of features = ', nf)
            new_vae_features_train = random_forest_feasture_selection(vae_features_train, y_train, nf)
            new_vae_features_test = random_forest_feasture_selection(vae_features_test, y_test, nf)
            new_lda_features_train = random_forest_feasture_selection(lda_features_train, y_train, nf)
            new_lda_features_test = random_forest_feasture_selection(lda_features_test, y_test, nf)
            new_data3_train = random_forest_feasture_selection(data3_train, y_train, nf)
            new_data3_test = random_forest_feasture_selection(data3_test, y_test, nf)

            data_tr = [new_vae_features_train, new_lda_features_train, new_data3_train]
            data_te = [new_vae_features_test, new_lda_features_test, new_data3_test]

            for c in range(len(classifiers)):
                classifier_name = classifiers[c]
                classifier = classifiers_funcs[c]
                print(classifier_name)
                for d in range(len(data_tr)):
                    this_tr = data_tr[d]
                    this_te = data_te[d]
                    this_feature = features[d]
                    print(this_feature)
                    tr, te = classifier(this_tr, this_te, y_train, y_test)
                    accuracy_tr, precision_tr, recall_tr, fscore_tr, fpr_tr, fnr_tr = tr
                    accuracy_te, precision_te, recall_te, fscore_te, fpr_te, fnr_te = te
                    results_pd.loc[ct, :] = accuracy_tr, classifier_name, this_feature, 'Train', 'Accuracy', nf
                    results_pd.loc[ct+1, :] = precision_tr, classifier_name, this_feature, 'Train', 'Precision', nf
                    results_pd.loc[ct+2, :] = recall_tr, classifier_name, this_feature, 'Train', 'Recall', nf
                    results_pd.loc[ct+3, :] = fscore_tr, classifier_name, this_feature, 'Train', 'F-score', nf
                    results_pd.loc[ct+4, :] = fpr_tr, classifier_name, this_feature, 'Train', 'FPR', nf
                    results_pd.loc[ct+5, :] = fnr_tr, classifier_name, this_feature, 'Train', 'FNR', nf
                    results_pd.loc[ct+6, :] = accuracy_te, classifier_name, this_feature, 'Test', 'Accuracy', nf
                    results_pd.loc[ct+7, :] = precision_te, classifier_name, this_feature, 'Test', 'Precision', nf
                    results_pd.loc[ct+8, :] = recall_te, classifier_name, this_feature, 'Test', 'Recall', nf
                    results_pd.loc[ct+9, :] = fscore_te, classifier_name, this_feature, 'Test', 'F-score', nf
                    results_pd.loc[ct+10, :] = fpr_te, classifier_name, this_feature, 'Test', 'FPR', nf
                    results_pd.loc[ct+11, :] = fnr_te, classifier_name, this_feature, 'Test', 'FNR', nf
                    ct += 12
                results_pd.to_csv(results_path + model_name + '_with_randoem_forest_fs.csv', index=False)


def evaluation_expermient(info):
    plot_pca_tsne(info)
    make_classifiers_df(info)
    plot_classifiers_result(info)

    make_classifiers_with_fs_chi2_df(info)
    plot_classifiers_with_fs_result(info, 'chi2')

    make_classifiers_with_fs_gini_df(info)
    plot_classifiers_with_fs_result(info, 'gini')


if __name__ == '__main__':

    if '-f' in sys.argv:
        top_folder = sys.argv[sys.argv.index('-f') + 1]
    else:
        top_folder = 'runs/'

    if '-d' in sys.argv:
        dataset_name = sys.argv[sys.argv.index('-d') + 1]
    else:
        dataset_name = exit('Error: You need to specify the dataset name with -d command. \nDatasets choices could be '
                            'Twitter or ISOT.')
        # dataset_name = 'ISOT'

    if '-a' in sys.argv:
        dataset_address = sys.argv[sys.argv.index('-a') + 1]
    else:
        dataset_address = exit('Error: You need to specify the address of top folder contatining both dataset folders '
                               'with -a command, eg. -a "data/".')
        # dataset_address = 'data/'

    if '-e' in sys.argv:
        epoch_no = int(sys.argv[sys.argv.index('-e') + 1])
    else:
        epoch_no = 1

    if '-t' in sys.argv:
        n_topics = int(sys.argv[sys.argv.index('-t') + 1])
    else:
        n_topics = 10

    if '-i' in sys.argv:
        n_iter = int(sys.argv[sys.argv.index('-i') + 1])

    else:
        n_iter = 1000

    if '-l' in sys.argv:
        latent_dim = int(sys.argv[sys.argv.index('-l') + 1])
    else:
        latent_dim = 32

    if '-w' in sys.argv:
        word2vec_dim = int(sys.argv[sys.argv.index('-w') + 1])
    else:
        word2vec_dim = 32

    run_info = make_run_info(top_folder, dataset_name, latent_dim, epoch_no, n_topics, n_iter, word2vec_dim)
    main_folder = run_info['main_folder']

    # data preparation
    if not os.path.exists(top_folder + 'data/' + dataset_name + '/x_train_' + str(word2vec_dim) + '.csv') or \
            not os.path.exists(top_folder + 'data/' + dataset_name + '/x_test_' + str(word2vec_dim) + '.csv'):
        prepare_data(run_info, top_folder, dataset_address)

    # VAE
    if not os.path.exists(main_folder + 'features/vae_train.npy') or \
            not os.path.exists(main_folder + 'features/vae_test.npy'):
        vae_experiment(run_info, top_folder)

    # LDA
    if not os.path.exists(main_folder + 'features/lda_tf_train.npy') or \
            not os.path.exists(main_folder + 'features/lda_tf_test.npy'):
        lda_experiment(run_info, top_folder)

    # LVAE (VAE + LDA)
    if not os.path.exists(main_folder + 'features/lvae_train_vae_lda_tf.npy') or \
            not os.path.exists(main_folder + 'features/lvae_test_vae_lda_tf.npy'):
        concat_features(run_info)

    # classification and evaluation
    evaluation_expermient(run_info)

    # plots_path = top_folder + 'Plots/'
    # if not os.path.exists(plots_path):
    #     os.makedirs(plots_path)
    #
    # results_path = top_folder + 'Results/'
    # if not os.path.exists(results_path):
    #     os.makedirs(results_path)

    # plot_all_pca_tsne_datasets(top_folder, plots_path, 'ISOT')
    # plot_all_pca_tsne_datasets(top_folder, plots_path, 'Twitter')
    # plot_all_pca_tsne_datasets(top_folder, plots_path, 'test')

    # make_results_df(top_folder, results_path, 'ISOT')
    # make_results_df(top_folder, results_path, 'test')

    # plot_accuracy_metrics(results_path, plots_path)
    #
    # make_results_with_fs_df(top_folder, results_path, 'ISOT')
    #
    # plot_accuracy_with_fs(results_path, plots_path)
    #
    # plot_history_independent(top_folder, plots_path, 'Twitter')
    # plot_history_independent(top_folder, plots_path, 'ISOT')

