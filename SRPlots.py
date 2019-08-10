import matplotlib.pyplot as plt
import pickle
import sys

historyfile = sys.argv[1]

#dataFile = 'SpeechRecog.npy'
data = pickle.load(open(historyfile, "rb"))

plt.figure('loss')
plt.plot(data['loss'])
plt.plot(data['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.yscale('log')
plt.grid(True)

plt.figure('Accuracy')
plt.plot(data['acc'])
plt.plot(data['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.yscale('log')
plt.grid(True)

plt.show()
