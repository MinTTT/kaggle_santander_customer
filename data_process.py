#%%
import numpy as np
from tqdm import tqdm
import pandas as pd


#%%
def count_number(data):
    count_data = []
    for var in tqdm(range(data.shape[1])):
        _, count_ = np.unique(data[:, var], return_counts=True)
        count_data = np.append(count_data, count_)
    return count_data


def find_unique_data(train):
    unique_samples = []
    unique_counts = np.zeros(np.shape(train))
    print('Now, we are identifying the unique rows: ')
    for var in tqdm(range(train.shape[1])):
        _, index_, counts_ = np.unique(train[:, var], return_index=True,
                                       return_counts=True)
        unique_counts[index_[counts_ == 1], var] += 1

    real_sample_index = np.argwhere(np.sum(unique_counts, axis=1)
                                    > 0)[:, 0]
    synthetic_sample_index = np.argwhere(np.sum(unique_counts, axis=1)
                                         == 0)[:, 0]
    print('Real sample number: ', real_sample_index.shape[0])
    print('Synthetic sample number:', synthetic_sample_index.shape[0])
    return real_sample_index, synthetic_sample_index


if __name__ == '__main__':
    print('Exit')

    #%%
    test = pd.read_csv(r'./test.csv')
    train = pd.read_csv(r'./train.csv')
    #%%
    real_inx, syn_inx = find_unique_data(test.iloc[:, 1:].values)
    #%%
    test_data = test.iloc[:, 1:-1].values
    unique_samples = []
    unique_counts = np.zeros(np.shape(test_data))

    for var in tqdm(range(test_data.shape[1])):
        _, index_, counts_ = np.unique(test_data[:, var], return_index=True,
                                       return_counts=True)
        unique_counts[index_[counts_ == 1], var] += 1

    #%%
    real_sample_index = np.argwhere(np.sum(unique_counts, axis=1)
                                    > 0)[:, 0]
    synthetic_sample_index = np.argwhere(np.sum(unique_counts, axis=1)
                                         == 0)[:, 0]
    print('Real sample number: ', real_sample_index.shape[0])
    print('Synthetic sample number:', synthetic_sample_index.shape[0])

    #%%
    unique_samples = test_data[real_sample_index, :].copy()
    generator = []

    # When test the code, I only use 1000 synthetic samples
    for cur_index in tqdm(synthetic_sample_index[:999]):
        cur_sample = test_data[cur_index]
        poten_generator = unique_samples == cur_sample
        feature_mask = np.sum(poten_generator, axis=0) == 1
        # Find the row index of generator
        verified_generator_mask = np.any(poten_generator[:, feature_mask], axis=1)
        # real site in real_sample_index
        verified_generator_for_sample = real_sample_index[
            verified_generator_mask]
        generator.append(set(verified_generator_for_sample))
    #%% test
    public_LB = generator[0]
    for x in tqdm(generator):
        if public_LB.intersection(x):
            public_LB = public_LB.union(x)

    private_LB = generator[1]
    for x in tqdm(generator):
        if private_LB.intersection(x):
            private_LB = private_LB.union(x)

    print(len(public_LB))
    print(len(private_LB))

    #%%

    train_data = train.iloc[:, 2:-1].values
    print('train data shape:', train_data.shape)
    print('test data shape:', test_data.shape)

    #%%




    count_train = count_number(train_data)
    count_test = count_number(test_data)
    #%%
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig_1 = plt.figure(figsize=(8, 8))
    sns.distplot(count_train, bins=50, label='Train Data', kde=False)
    sns.distplot(count_test, bins=50, label='Test Data', kde=False)
    plt.yscale('log')
    plt.legend()
    plt.show()

    #%%
