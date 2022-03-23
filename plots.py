import sys
sys.path.append('../')
import matplotlib
matplotlib.use('Agg')
from feature_selection import *
import seaborn as sns
import pickle as pkl


def plot_history_independent(top_folder, plots_path, dataset_name):
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[1]

        with open(run + '/model_history.pickle', 'rb') as handle:
            model_history = pkl.load(handle)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        axes[0].plot(model_history['decoded_txt_accuracy'], label='Reconstruction (Training)')
        axes[0].plot(model_history['fnd_output_accuracy'], label='Classifier (Training)')
        axes[0].plot(model_history['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
        axes[0].plot(model_history['val_fnd_output_accuracy'], label='Classifier (Validation)')
        # axes[0].set_title('Subplot 1', fontsize=14)
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Accuracy', fontsize=14)
        axes[0].legend(loc='lower right')

        axes[1].plot(model_history['decoded_txt_loss'], label='Reconstruction (Training)')
        axes[1].plot(model_history['fnd_output_loss'], label='Classifier (Training)')
        axes[1].plot(model_history['val_decoded_txt_loss'], label='Reconstruction (Validation)')
        axes[1].plot(model_history['val_fnd_output_loss'], label='Classifier (Validation)')
        # axes[0].set_title('Subplot 1', fontsize=14)
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('Loss', fontsize=14)
        axes[1].legend(loc='lower right')

        # Save figure
        fig.savefig(plots_path + 'model_history' + model_name + '.png')
        fig.savefig(plots_path + 'model_history' + model_name + '.pdf')


def plot_history_with_2_outputs(main_folder, model_history):

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    axes[0].plot(model_history['decoded_txt_accuracy'], label='Reconstruction (Training)')
    axes[0].plot(model_history['fnd_output_accuracy'], label='Classifier (Training)')
    axes[0].plot(model_history['val_decoded_txt_accuracy'], label='Reconstruction (Validation)')
    axes[0].plot(model_history['val_fnd_output_accuracy'], label='Classifier (Validation)')
    # axes[0].set_title('Subplot 1', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].legend(loc='lower right')

    axes[1].plot(model_history['decoded_txt_loss'], label='Reconstruction (Training)')
    axes[1].plot(model_history['fnd_output_loss'], label='Classifier (Training)')
    axes[1].plot(model_history['val_decoded_txt_loss'], label='Reconstruction (Validation)')
    axes[1].plot(model_history['val_fnd_output_loss'], label='Classifier (Validation)')
    # axes[0].set_title('Subplot 1', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Loss', fontsize=14)
    axes[1].legend(loc='lower right')

    # Save figure
    fig.savefig(main_folder + 'model_history.png')
    fig.savefig(main_folder + 'model_history.pdf')


def general_plot():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    x = np.linspace(1, 100, 100)
    y1 = (1+(1/x))**x
    y2 = (1-(1/x))**x
    axes[0].plot(x, y1, label=r'$f_1(x) = (1 + \frac{1}{x})^x$')
    axes[0].set_xlabel('x', fontsize=14)
    axes[0].set_ylabel(r'$f_1(x) = (1 + \frac{1}{x})^x$', fontsize=10)
    axes[0].legend(loc='upper left')

    axes[1].plot(x, y2, label=r'$f_1(x) = (1 - \frac{1}{x})^x$')
    axes[1].set_xlabel('x', fontsize=14)
    axes[1].set_ylabel(r'$f_1(x) = (1 - \frac{1}{x})^x$', fontsize=10)
    axes[1].legend(loc='upper right')
    plt.show()

    plt.subplots(figsize=(10, 10))
    x = np.linspace(1, 100, 100)
    y1 = (1+(1/x))**x
    y2 = (1-(1/x))**x
    y3 = y2*y1
    plt.plot(x, y1, label=r'$f(x) = (1 + \frac{1}{x})^x$')
    plt.plot(x, y2, label=r'$f(x) = (1 - \frac{1}{x})^x$')
    plt.plot(x, y3, label=r'$f(x) = (1 + \frac{1}{x})^x (1 - \frac{1}{x})^x$')
    plt.xlabel('x', fontsize=14)
    plt.ylabel(r'$f(x)$', fontsize=10)
    plt.legend(loc='upper right')
    plt.show()


def plot_3_pca(data1, data2, data3, y, folder, plot_name='self'):

    pca1_data1, pca2_data1 = make_pca(data1, n_components=2)
    pca1_data2, pca2_data2 = make_pca(data2, n_components=2)
    pca1_data3, pca2_data3 = make_pca(data3, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    axes[0].scatter(pca1_data1[list(np.where(y == 0)[0])], pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0].scatter(pca1_data1[list(np.where(y == 1)[0])], pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[1].scatter(pca1_data2[list(np.where(y == 0)[0])], pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1].scatter(pca1_data2[list(np.where(y == 1)[0])], pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[2].scatter(pca1_data3[list(np.where(y == 0)[0])], pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2].scatter(pca1_data3[list(np.where(y == 1)[0])], pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    axes[2].legend(loc='upper right')
    axes[0].set_ylabel('PCA2', fontsize=8)
    axes[1].set_ylabel('PCA2', fontsize=8)
    axes[2].set_xlabel('PCA1', fontsize=8)
    axes[2].set_ylabel('PCA2', fontsize=8)
    axes[0].set_title('VAE features', fontsize=10)
    axes[1].set_title('LDA features', fontsize=10)
    axes[2].set_title('VAE + LDA features', fontsize=10)
    plt.tight_layout()
    fig.savefig(folder + 'PCA_all_' + plot_name + '.png')
    fig.savefig(folder + 'PCA_all_' + plot_name + '.pdf')


def plot_3_tsne(data1, data2, data3, y, folder, plot_name='self'):

    pca1_data1, pca2_data1 = make_tSNE(data1, tsne_perplexity=40, n_components=2)
    pca1_data2, pca2_data2 = make_tSNE(data2, tsne_perplexity=40, n_components=2)
    pca1_data3, pca2_data3 = make_tSNE(data3, tsne_perplexity=40, n_components=2)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    axes[0].scatter(pca1_data1[list(np.where(y == 0)[0])], pca2_data1[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[0].scatter(pca1_data1[list(np.where(y == 1)[0])], pca2_data1[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[1].scatter(pca1_data2[list(np.where(y == 0)[0])], pca2_data2[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[1].scatter(pca1_data2[list(np.where(y == 1)[0])], pca2_data2[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[2].scatter(pca1_data3[list(np.where(y == 0)[0])], pca2_data3[list(np.where(y == 0)[0])], s=5, alpha=0.5, label='real')
    axes[2].scatter(pca1_data3[list(np.where(y == 1)[0])], pca2_data3[list(np.where(y == 1)[0])], s=5, alpha=0.5, label='fake')
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    axes[2].legend(loc='upper right')
    axes[0].set_ylabel('tSNE2', fontsize=8)
    axes[1].set_ylabel('tSNE2', fontsize=8)
    axes[2].set_xlabel('tSNE1', fontsize=8)
    axes[2].set_ylabel('tSNE2', fontsize=8)
    axes[0].set_title('VAE features', fontsize=10)
    axes[1].set_title('LDA features', fontsize=10)
    axes[2].set_title('VAE + LDA features', fontsize=10)
    plt.tight_layout()
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.png')
    fig.savefig(folder + 'tSNE_all_' + plot_name + '.pdf')


def plot_all_pca_tsne_datasets(top_folder, plots_path, dataset_name):
    runs = [os.path.join(top_folder, dname) for dname in os.listdir(top_folder) if dataset_name in dname]
    for run in runs:
        model_name = run.split('/')[1]
        ntopics = int(model_name.split('_')[model_name.split('_').index('topics')+1])
        nlatent = int(model_name.split('_')[model_name.split('_').index('dim')+1])
        w2v = int(model_name.split('_')[model_name.split('_').index('w2v')+1])
        features_path = run + '/features/'
        vae_features_train = np.load(features_path + 'vae_train.npy')
        vae_features_test = np.load(features_path + 'vae_test.npy')
        lda_features_train = np.load(features_path + 'lda_tf_train.npy')
        lda_features_test = np.load(features_path + 'lda_tf_test.npy')
        data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
        data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
        y_train = np.load(run + '/train_label.npy')
        y_test = np.load(run + '/test_label.npy')

        plot_name = dataset_name + '_n_topis_' + str(ntopics) +'_latent_dim_' + str(nlatent) + '_w2v_' + str(w2v) + '_train'
        plot_3_tsne(vae_features_train, lda_features_train, data3_train, y_train, plots_path, plot_name=plot_name)
        plot_3_pca(vae_features_train, lda_features_train, data3_train, y_train, plots_path, plot_name=plot_name)

        plot_name = dataset_name + '_n_topis_' + str(ntopics) +'_latent_dim_' + str(nlatent) + '_w2v_' + str(w2v) + '_test'
        plot_3_tsne(vae_features_test, lda_features_test, data3_test, y_test, plots_path, plot_name=plot_name)
        plot_3_pca(vae_features_test, lda_features_test, data3_test, y_test, plots_path, plot_name=plot_name)


def plot_pca_tsne(run_info):

    main_folder = run_info['main_folder']
    dataset_name = run_info['dataset_name']
    latent_dim = run_info['latent_dim']
    word2vec_dim = run_info['word2vec_dim']
    n_topics = run_info['n_topics']

    features_path = main_folder + 'features/'
    vae_features_train = np.load(features_path + 'vae_train.npy')
    vae_features_test = np.load(features_path + 'vae_test.npy')
    lda_features_train = np.load(features_path + 'lda_tf_train.npy')
    lda_features_test = np.load(features_path + 'lda_tf_test.npy')
    data3_train = np.concatenate((vae_features_train, lda_features_train), axis=1)
    data3_test = np.concatenate((vae_features_test, lda_features_test), axis=1)
    y_train = np.load(main_folder + 'train_label.npy')
    y_test = np.load(main_folder + 'test_label.npy')

    plot_name_tr = dataset_name + '_n_topis_' + str(n_topics) +'_latent_dim_' + str(latent_dim) + '_w2v_' + \
                   str(word2vec_dim) + '_train'
    plot_3_tsne(vae_features_train, lda_features_train, data3_train, y_train, main_folder, plot_name=plot_name_tr)
    plot_3_pca(vae_features_train, lda_features_train, data3_train, y_train, main_folder, plot_name=plot_name_tr)

    plot_name_te = dataset_name + '_n_topis_' + str(n_topics) +'_latent_dim_' + str(latent_dim) + '_w2v_' + \
                   str(word2vec_dim) + '_test'
    plot_3_tsne(vae_features_test, lda_features_test, data3_test, y_test, main_folder, plot_name=plot_name_te)
    plot_3_pca(vae_features_test, lda_features_test, data3_test, y_test, main_folder, plot_name=plot_name_te)


def plot_accuracy_metrics(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if '.csv' in dname and 'with_fs'
            not in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[2].split('.csv')[0]
        model_df = pd.read_csv(run)
        model_df_filtered = model_df[model_df['Metric'] != 'FNR']
        model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'FPR']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM_nonlinear']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
        model_df_filtered.loc[model_df_filtered['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'

        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()
        sns_plot = sns.relplot(data=model_df_filtered, x="Metric", y="Value", col="Feature", hue="Classifier", style="Dataset", aspect=.45)
        leg = sns_plot._legend
        leg.set_bbox_to_anchor([1.15, 0.4])
        plt.tight_layout()
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.png')
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.pdf')
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()


def plot_classifiers_result(run_info):
    main_folder = run_info['main_folder']
    model_df = pd.read_csv(main_folder + 'Classifiers.csv')
    model_df_filtered = model_df[model_df['Metric'] != 'FNR']
    model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM_nonlinear']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
    model_df_filtered.loc[model_df_filtered['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'

    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()
    sns_plot = sns.relplot(data=model_df_filtered, x="Metric", y="Value", col="Feature", hue="Classifier",
                           style="Dataset", aspect=.45)
    leg = sns_plot._legend
    leg.set_bbox_to_anchor([1.15, 0.4])
    plt.tight_layout()
    sns_plot.savefig(main_folder + 'Metrics_classifiers.png')
    sns_plot.savefig(main_folder + 'Metrics_classifiers.pdf')
    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()


def plot_accuracy_metrics2(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if '.csv' in dname and 'with_fs'
            not in dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[2].split('.csv')[0]
        model_df = pd.read_csv(run)
        model_df_filtered = model_df[model_df['Metric'] != 'FNR']
        model_df_filtered = model_df_filtered[model_df['Metric'] != 'FPR']
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()
        sns_plot = sns.relplot(data=model_df_filtered, x="Classifier", y="Value", col="Feature", hue="Metric", style="Dataset")
        plt.tight_layout()
        sns_plot.savefig(plots_path + 'Accuracy_2_' + model_name + '.png')
        sns_plot.savefig(plots_path + 'Accuracy_2_' + model_name + '.pdf')
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()


def plot_accuracy_with_fs(results_path, plots_path):
    runs = [os.path.join(results_path, dname) for dname in os.listdir(results_path) if '.csv' in dname and 'with_fs' in
            dname]
    for run in runs:
        print(run)
        model_name = run.split('/')[2].split('.csv')[0]
        model_df = pd.read_csv(run)
        model_df_filtered = model_df[model_df['Metric'] != 'FNR']
        model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM (nonlinear)']
        model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
        model_df_filtered = model_df_filtered[model_df_filtered['Dataset'] != 'Train']
        model_df_filtered = model_df_filtered[model_df_filtered['Metric'] == 'Accuracy']
        # model_df_filtered.loc[model_df_filtered['Classifier'] == 'SVM_linear', 'Classifier'] = 'SVM (linear)'

        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()

        # sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", row="Feature",
        #                        col="Metric", hue="Classifier", style="Dataset", aspect=1.0, kind='line')
        sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", col="Feature",
                               hue="Classifier", aspect=5.0, kind="line")
        # sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", col="Classifier",
        #                        hue="Feature", style="Dataset", aspect=1.0, kind='line', col_wrap=3)

        leg = sns_plot._legend
        leg.set_bbox_to_anchor([1, 0.45])
        plt.tight_layout()
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.png')
        sns_plot.savefig(plots_path + 'Accuracy_' + model_name + '.pdf')
        plt.close('all')
        plt.close()
        plt.clf()
        plt.cla()


def plot_classifiers_with_fs_result(run_info, fs_name):
    main_folder = run_info['main_folder']
    model_df = pd.read_csv(main_folder + 'Classifiers_with_' + fs_name + '_fs.csv')

    model_df_filtered = model_df[model_df['Metric'] != 'FNR']
    model_df_filtered = model_df_filtered[model_df_filtered['Metric'] != 'FPR']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'SVM (nonlinear)']
    model_df_filtered = model_df_filtered[model_df_filtered['Classifier'] != 'linear disriminat Analysis']
    model_df_filtered = model_df_filtered[model_df_filtered['Dataset'] != 'Train']
    model_df_filtered = model_df_filtered[model_df_filtered['Metric'] == 'Accuracy']

    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()

    # sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", row="Feature",
    #                        col="Metric", hue="Classifier", style="Dataset", aspect=1.0, kind='line')
    sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", col="Feature",
                           hue="Classifier", aspect=5.0, kind="line")
    # sns_plot = sns.relplot(data=model_df_filtered, x="# Selected Features", y="Value", col="Classifier",
    #                        hue="Feature", style="Dataset", aspect=1.0, kind='line', col_wrap=3)

    leg = sns_plot._legend
    leg.set_bbox_to_anchor([1, 0.45])
    plt.tight_layout()
    sns_plot.savefig(main_folder + 'Metrics_classifiers_with_' + fs_name + '_fs.png')
    sns_plot.savefig(main_folder + 'Metrics_classifiers_with_' + fs_name + '_fs.pdf')
    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()

