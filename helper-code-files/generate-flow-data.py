import os
import subprocess

dataDirectory = '/home/data/CPTC'

def main():

	for dirName, subdirList, fileList in os.walk(dataDirectory):
		i = 1
		if (len(fileList) != 0):
			for pcap in fileList:
				runPktFlow = './pkt2flow/pkt2flow -v -o data-flows/' + \
							 dirName.split('/')[4] + '-' + str(i) + '/ ' + dirName + '/' + pcap
				output=subprocess.check_output(['bash', '-c', runPktFlow])
				i += 1
main()	
