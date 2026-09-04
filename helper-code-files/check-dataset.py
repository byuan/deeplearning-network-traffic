from collections import defaultdict
import pandas as pd
import numpy as np

def main():
	df = pd.read_csv('datafiles/dataset.csv')
	d = defaultdict(int)
	for protocol in df.values:
		d[protocol[0]] += 1
	print(d.items())
	for item in d.items():
		print(item[0],item[1])

main()	
