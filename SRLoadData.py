from SRData import load_dataset_from_hdf5

datasetfile = 'SRData.h5'

# Loads dataset from disk
x_train, y_train = load_dataset_from_hdf5(datasetfile)

print(f'Training vector: {x_train.shape} --> (Num Examples, Num Time Steps, Features)')
print(f'Target vector: {y_train.shape} --> (Num Examples, Word Number)' )
