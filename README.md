# LVAE: A framework for Fake News Detection by Variational Autoencoder and Topic Modelling

With the growth of social media and User Generated Content in recent decade, people are more exposed to news with unknown sources and fake content. Validating the credibility of such information, however, is not a trivial task for majority of the users and spreading false information could potentially lead to losses and crimes. As a result, it is essential to develop accurate techniques for distinguishing fake and real news. Due to the absence of multimedia and information about the author and spread pattern in many real-world sources of news, we focus on extracting relevant features only from the textual content. Our contribution is proposing a multi-modal approach that aggregates the hidden representation of textual news using a variational autoencoder and topic-related features inferred from Latent Dirichlet Allocation (LDA) mixture model to achieve a more accurate and interpretable model.

**Requirements Installation Guide**

Python 3.6

\# gensim (3.8.3) or older

conda install -c anaconda gensim

\# tensorflow (2.4.1)

conda create -n tf tensorflow

conda activate tf

\#or 

conda create -n tf-gpu tensorflow-gpu

conda activate tf-gpu

\# keras (2.4.3)

conda install -c conda-forge keras

\# wordcloud

conda install -c conda-forge wordcloud

\# nltk 3.5

conda install -c anaconda nltk

\# sklearn

pip3 install -U scikit-learn

\# langdetect (1.0.8)

pip3 install --user langdetect

\# pandas

conda install -c anaconda pandas

**Command line**

python3 main.py -f <main folder address for saving the variables> -d <dataset name: 'Twitter' or 'ISOT'> -a <top folder of dataset address containing ISOT and Twitter folders> -e <#epochs> -t <#topics> -i <#iterations> -l <#latent features>

-f and -d and -a are mandatory
