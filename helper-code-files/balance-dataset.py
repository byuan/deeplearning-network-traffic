from collections import defaultdict
import pandas as pd
import numpy as np

MAX_CLASS_TYPE = 2500

def main():
	df = pd.read_csv('datafiles/dataset.csv')
	d = defaultdict(int)
	for protocol in df.values:
		d[protocol[0]] += 1
	print(d.items())
	data_list = df.as_matrix(columns=None)
	update_list = []
	for item in d.items():
		if (item[1] > 3000):
			print(item[0])
			count = 0 
			for i in range(len(data_list)):
				if (count < MAX_CLASS_TYPE):
					if (item[0] == data_list[i][0]):
						row = data_list[i]
						update_list.append(row)
						count += 1
		elif (item[1] > 400):
			print(item[0])
			for i in range(len(data_list)):
				if (item[0] == data_list[i][0]):
					row = data_list[i]
					update_list.append(row)

	np_data_list = np.asarray(update_list)
	np.random.shuffle(np_data_list)
	dataset = pd.DataFrame(np_data_list)
	dataset.to_csv('dataset.csv', index=False, header=False)	

	print(dataset.shape)
	
main()	
