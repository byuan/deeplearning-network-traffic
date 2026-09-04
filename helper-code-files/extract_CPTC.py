import os
import subprocess
import pandas as pd
import timeit

start = timeit.default_timer()

dataFlows = "/home/avj2668/deeplearning-network-traffic/data-flows"
openfile = "ndpi-output.txt"
dataList = []

def main():
	for dirName, subdirList, fileList in os.walk(dataFlows):	
		if (len(fileList) != 0 ):	
			if ('day1' in dirName.split('/')[5])
				print(dirName, len(fileList))
				for fname in fileList:
					output = runBashCommand(dirName, fname)							# run bash command for nDPI
					identifiedProtocol, payloadSize = readProtocolInfo(openfile)	# open file to read detected protocol info
					if (identifiedProtocol is not None):
						createDataset(identifiedProtocol, dataFlows, dirName, fname, dataList)
	dataset = pd.DataFrame(dataList)
	dataset.to_csv('dataset-day1-part3.csv', index = False, header = False)
	print(dataset)

	stop = timeit.default_timer()
	print (stop - start)					

def runBashCommand(dirName, fname):

	bashCmd = "./nDPI/example/ndpiReader -i " + dirName + "/" + fname + " -w " + openfile
	output = subprocess.check_output(['bash', '-c', bashCmd])
	return output

def readProtocolInfo(openfile):

	file = open(openfile, "r")
	identifiedProtocol = None
	for protocol in file:
		payloadSize = protocol.split("\t")[2]
		if (int(payloadSize) >= 1000):
			identifiedProtocol = protocol.split("\t")[0]
	return identifiedProtocol, payloadSize

def createDataset(identifiedProtocol, flowDir, dirName, fname, dataList):
	mkdir = "mkdir " + flowDir + "/flows"
	subprocess.check_output(['bash', '-c', mkdir])
	tcpflow = "tcpflow -b 1024 " + "-o " + flowDir + "/flows -r " + dirName + "/" + fname
	subprocess.check_output(['bash', '-c', tcpflow])
	flows = flowDir + "/flows"
	for dirN, subdir, flowList in os.walk(flows):
		for flow in flowList:
			if (flow != "report.xml"):
				getFlowSize = "xxd -c 1 -p " + flows + "/" + flow + " | wc -l"
				flowSize = subprocess.check_output(['bash', '-c', getFlowSize])
				if (int(flowSize) == 1024):
					getHex = "xxd -c 1 -p " + flows + "/" + flow
					hexOfFlow = subprocess.check_output(['bash', '-c', getHex])
					hexList = hexOfFlow.split()
					flowInt = [int(x, 16) for x in hexList]
					flowInt.insert(0, identifiedProtocol)
					dataList.insert(0, flowInt)
	rmdirCmd = "rm -r " + flowDir + "/flows"
	subprocess.check_output(['bash', '-c', rmdirCmd])

main()
