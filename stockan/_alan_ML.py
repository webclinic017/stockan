#!/usr/bin/env python
# coding: utf-8

# # stock_analysis_with_machine_learning_rf
# Machine Learning KNN/RandomForest in Finacial Application
# stock return forecasts with technical indicatiors via ML

__author__ = "Ted Hong"
__email__ = "ted@beyondbond.com"
__license__ = "MIT"
__version__ = "0.0.1"
__copyright__ = "Copyright 2019, Ted Hong"


# ## 1. Request and Clean Data 
# #### Using U.S. EST time as local timestamp to pull SPY minute data
# Suggested tickers to use from DOW30:  MMM, AXP, AAPL, BA, CAT, CVX, CSCO, KO, DWDP, XOM, GS, HD, IBM, INTC, JNJ, JPM, MCD, MRK, MSFT, NKE, PFE, PG, TRV, UNH, UTX, VZ, V, WMT, WBA, DIS
# 
# get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import json
from optparse import OptionParser
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import requests
import os, time
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
#from sklearn.svm import LinearSVC, SVC
#from sklearn.neural_network import MLPClassifier

# Set local time zone to NYC
os.environ['TZ']='America/New_York'
time.tzset()
t=time.localtime() # string
print('Run analysis @ {}'.format(time.ctime()))

def cumsum_clip(a,xfl=-99,xcap=99):
	'''
	calc cumulative sum of array a[] clip to [xfl,xcap]
	'''
	b=[]
	tx=0
	for x in a:
		tx = np.clip(tx+x,xfl,xcap)
		b.append(tx)
	return np.array(b)

def pullStockPrice(ticker,nday):
	'''
	Pull stock prices from last 7 days to the beginning of today
	'''
	urx="https://query1.finance.yahoo.com/v8/finance/chart/{}?region=US&lang=en-US&includePrePost=false&interval={}&period1={}&period2={}"
	period1=(datetime.datetime.today().date()-datetime.timedelta(nday)).strftime('%s')
	period2=datetime.datetime.today().date().strftime('%s')
	url = urx.format(ticker,'1m',period1,period2)

	jTmp = pd.read_json(url)['chart']['result'][0]
	pbdatetime = [ datetime.datetime.fromtimestamp(int(x)) for x in jTmp['timestamp'] ]
	df=pd.DataFrame(jTmp['indicators']['quote'][0])
	df.loc[:,'ticker']=ticker

	# use numerical index instead of time index for better display multiple days plot
	#df.set_index(pd.DatetimeIndex(pbdatetime),inplace=True)

	df.dropna(inplace=True)
	df = df[['open','high','low','close']]
	return df,pbdatetime

def plotStockPrice(df,ticker,pbdatetime):
	title = '{} asof {}'.format(ticker,pbdatetime[-1])
	fig, ax=plt.subplots(figsize=(12,6))
	df.plot(ax=ax,title=title)

	# set xticks for better display multiple days plot
	plt.locator_params(axis='x', nbins=20)  # x-axis
	vn=range(len(pbdatetime))
	xtcks = [pbdatetime[int(j)].strftime('%m/%d-%H:%M') if j in vn else None for j in ax.get_xticks()]
	ax.set_xticklabels(xtcks)

	plt.xticks(rotation='20',fontsize=10)
	plt.show()
	return ax


# ## 2. Build Features and Labeling
# ### Use RSI, MACD, MA(10) and Bollinger Band and ADX as stock singals
# ### Use up/down/no_change for y labeling
# #### Note, install ta for technical indicators
# #### ! pip install ta
def buildTaFeatures(df):
	import ta
	# RSI
	df['rsi'] = ta.momentum.rsi(df['close'],n=14)
	df['rsi_singal'] = [ 1 if x>=70 else -1 if x<=30 else 0 for x in df['rsi'] ]
	#df['rrt'] = df['close'].pct_change()
	df['rrt'] = np.log(df['close']).diff()

	# MACD setup
	df['macd'] = ta.trend.macd(df['close'], n_fast=12, n_slow=26, fillna=False)
	df['macd_xover']=df['macd'] - df['macd'].ewm(span=9).mean()
	df['macd_xover_signal'] = [ 1 if x>0 else -1 if x<0 else 0 for x in df['macd_xover'] ]
	df['macd_signal'] = (np.sign(df['macd_xover_signal'] - df['macd_xover_signal'].shift(1)))

	# ADX setup
	df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], n=14).values

	# Bollinger Band Moving average setup
	df['bband_up'] = ta.volatility.bollinger_hband(df['rrt'], n=20, ndev=0.5)
	df['bband_dn'] = ta.volatility.bollinger_lband(df['rrt'], n=20, ndev=0.5)
	df['bband'] = ta.volatility.bollinger_mavg(df['rrt'], n=20)
	df['bband'] = df['bband'].shift(1)
	df = df.replace([np.inf, -np.inf], np.nan)
	return df

def set_ylabel_forecast(df, fcs_prd=1):
	# shift rate-of-return 1-period earlier for next period prediction
	ystd = df['rrt'][:-120].std()
	df['rrt']=df['rrt'].shift(-fcs_prd)

	# set y labeling (5-zones labeling)
	# df['Direction'] = [ 1 if x>=0 else -1 for x in df['rrt'] ]
	df['Direction'] = [ 2 if x>1.0*ystd else 1 if x>0.5*ystd else 0 if x >-1*ystd else -1 if x>-2*ystd else -2 for x in df['rrt'] ]
	df.dropna(inplace=True)
	return df

def dataManipulation(df,test_size=0.2,feature_list=[]):
	# ### Split Data into Train and Test Sets
	XData = df[feature_list].values
	yData = df['Direction'].values
	X_train_raw, X_test_raw, y_train, y_test = train_test_split(XData,yData,test_size=test_size,shuffle=False)
	# ###  Data Selection and Normalization
	scaler = MinMaxScaler(feature_range=(0, 1))
	X_train = scaler.fit(X_train_raw).transform(X_train_raw)
	X_test = scaler.fit(X_test_raw).transform(X_test_raw)
	return(X_train, X_test, y_train, y_test)

def trainAndPredict(clname,clf, X_train, X_test, y_train, y_test):
	# Train each of the models on the training set
	clf.fit(X_train, y_train)

	# Make an array of predictions on the test set
	fitted = clf.predict(X_train)
	pred = clf.predict(X_test)
	score = clf.score(X_test, y_test)
	#acScore = accuracy_score(y_test,pred)
	confMtx = confusion_matrix(y_test, pred)
	hits = confMtx.diagonal().sum()
	imft = {k:v for k,v in zip(feature_list,clf.feature_importances_)}

	# Output the hit-rate and the confusion matrix for each model
	#print("Method: {}, Score:{:.5f}, accuracyScore:{:.5f}".format(clname, score,acScore))
	print("Method: {}, Score:{:.5f}".format(clname, score))
	print("confusion Matrix:\n{}".format(confMtx) )  
	# list important features in order
	imfx = ['{}:\t{:.2f}'.format(k,imft[k]) for k in sorted(imft, key=imft.get,reverse=True)]
	print("=== Important Futures:\n {}".format('\n '.join(imfx)))
	return hits,score,confMtx,imft,fitted,pred

def pnlPlot(ticker,clname,df,pbdatetime,X_train,X_test,y_train,y_test,fitted,pred):
	''' Profit and Loss Analysis
	'''
	nobs=X_test.shape[0]
	nobt=y_train.shape[0]
	print(df.columns)
	print(df.tail())
	act_rrt = df['rrt'].values
	train_rrt = df['rrt'].iloc[:nobt].values
	fitted_rrt = fitted*y_train*train_rrt
	test_rrt = df['rrt'].iloc[-nobs:].values
	pred_rrt = pred*y_test*test_rrt
	fit_rrt = np.append(fitted_rrt,pred_rrt)
	fig, ax=plt.subplots(figsize=(11,6))
	dg = pd.DataFrame(np.transpose([fit_rrt,act_rrt]),columns=[clname,ticker])
	title='{} Return'.format(ticker)
	tidx=[x.strftime('%m/%d-%H:%M') for x in pbdatetime[:nobt+nobs]]
	dg.index = tidx
	dg.iloc[-300:].plot(ax=ax,title=title,ylim=(-0.001,0.001))
	plt.show()

def cumPnlPlot(ticker,clname,df,pbdatetime,X_train,X_test,y_train,y_test,fitted,pred):
	''' Cumulative Profit and Loss Analysis
	'''
	nobs=X_test.shape[0]
	test_rrt = df['rrt'].iloc[-nobs:].values
	#pred_rrt = pred*y_test*test_rrt
	pred = cumsum_clip(pred,-2,4)*0.5
	print >> sys.stderr, "Positions: {}".format([(x,round(100*y,2),round(x*y*100,2)) for x,y in zip(pred,test_rrt)])
	pred_rrt = pred*test_rrt
	dh = pd.DataFrame(np.transpose([pred_rrt,test_rrt]),columns=[clname,ticker])
	dh = dh.cumsum().apply(lambda x:np.exp(x)*100-100)
	dh.set_index(pd.DatetimeIndex(pbdatetime[-nobs:]),inplace=True)
	title = '{} Cumulative P/L Forecasts'.format(ticker)
	dh.plot(title=title)
	plt.legend()
	plt.xlabel(pbdatetime[-1].strftime('%Y-%m-%d'))
	plt.ylabel('Profit/Loss in %')
	plt.show()
	print("Job finished @ {}".format(datetime.datetime.now()))

def treeVisualization(X_train, y_train):
	''' Tree Visualization
	'''
	#! pip install pydot --user
	from sklearn import tree
	# Import tools needed for visualization
	from sklearn.tree import export_graphviz
	from sklearn.externals.six import StringIO
	import pydot
	from IPython.display import Image, display

	# Export the image to a dot file
	tree_clf = tree.DecisionTreeClassifier(max_depth=3
					      ).fit(X_train, y_train)
	dot_data = StringIO()
	export_graphviz(tree_clf, out_file=dot_data,feature_names=feature_list,rounded=True)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
	Image(graph.create_png(), width=1000)

def run_ML(ticker,nday=7,feature_list=['adx','rsi','macd','bband'],test_size=0.2):
	modelX= ("RF", RandomForestClassifier(
		n_estimators=1000, criterion='gini', max_depth=None,
		min_samples_split=10, min_samples_leaf=2, max_features='auto',
		bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=1)
	)
	#1. Pull stock prices from last 7 days to the beginning of today
	df,pbdatetime = pullStockPrice(ticker,nday)
	#ax = plotStockPrice(df,ticker,pbdatetime)
	#2. Build Features and Labeling
	df = buildTaFeatures(df)
	df = set_ylabel_forecast(df, fcs_prd=1)
	
	#3-4. Dataset Manipulation & Normalization
	X_train, X_test, y_train, y_test = dataManipulation(df,test_size,feature_list)
	#5. Fitting and Predicting
	clname, clf = modelX
	hits,score,confMtx,imft,fitted,pred = trainAndPredict(clname,clf,X_train, X_test, y_train, y_test)
	#6. Fitting and predition Analysis
	#pnlPlot(ticker,clname,df,pbdatetime,X_train,X_test,y_train,y_test,fitted,pred)
	#7. Profit and Loss Analysis
	#cumPnlPlot(ticker,clname,df,pbdatetime,X_train,X_test,y_train,y_test,fitted,pred)
	#8. Tree Visualization
	#treeVisualization(X_train, y_train)
	
	confMtxJs = json.dumps(confMtx.tolist())
	imftJs = json.dumps(imft)
	pbepoch = int(datetime.datetime.now().strftime('%s000'))
	rfc_id=pred[-1]
	clrLst=['red','yellow','green']
	rfc_color=clrLst[np.clip(rfc_id,-1,1)+1]
	da = dict(ticker=ticker,hit_ratio=score,hit_num=hits,rfc_id=pred[-1],rfc_color=rfc_color,comment=None,pbepoch=pbepoch)
	dfa = pd.DataFrame([da])[['ticker','hit_ratio','hit_num','rfc_id','rfc_color','comment','pbepoch']]
	db = dict(ticker=ticker,import_feature=imftJs,confusion_matrix=confMtxJs,algorithm=clname,pbepoch=pbepoch)
	dfb = pd.DataFrame([db])[['ticker','import_feature','confusion_matrix','algorithm','pbepoch']]
	return dfa, dfb

def opt_alan_calc(argv,retParser=False):
	parser = OptionParser(usage="usage: %prog [option] SYMBOL1 ...", version="%prog 0.71",
		description="ALAN ML Optimizer")
	parser.add_option("","--src",action="store",dest="src",default="tw",
		help="source [tw|yahoo|iex](default: tw)")
	parser.add_option("-s","--start",action="store",dest="start",
		help="start YYYY-MM-DD (default: 3-years-ago)")
	parser.add_option("-e","--end",action="store",dest="end",
		help="end YYYY-MM-DD (default: today)")
	parser.add_option("","--days",action="store",dest="nday",default=7,type=int,
		help="number of days from endDate (default: 7)")
	parser.add_option("-d","--database",action="store",dest="dbname",default="ara",
		help="database (default: ara)")
	parser.add_option("","--host",action="store",dest="hostname",default="localhost",
		help="db host (default: localhost)")
	parser.add_option("-t","--table",action="store",dest="tablename",default="ml_daily",
		help="db tablename for ohlc history (default: ml_daily)")
	parser.add_option("-w","--wmode",action="store",dest="wmode",default="replace",
		help="db table write-mode [replace|append] (default: replace)")
	parser.add_option("","--features",action="store",dest="features",
		help="list of features)")
	parser.add_option("","--no_database_save",action="store_false",dest="saveDB",default=True,
		help="no save to database (default: save to database)")
	parser.add_option("","--extra_js",action="store",dest="extraJS",
		help="extra JSON in DICT format.")
	parser.add_option("","--debug",action="store_true",dest="debugTF",default=False,
		help="debugging (default: False)")
	(options, args) = parser.parse_args(argv[1:])
	if retParser is True:
		return parser
	opts = vars(options)
	try:
		from _alan_str import extra_opts
		extra_opts(opts,xkey='extraJS',method='JS',updTF=True)
	except Exception as e:
		print(str(e))
	if 'sep' in opts:
		opts['sep']=opts['sep'].encode().decode('unicode_escape') if sys.version_info.major==3 else opts['sep'].decode('string_escape')
	return (opts, args)

if __name__ == '__main__':
	(opts, args)=opt_alan_calc(sys.argv)
	if len(args) == 0:
		print >> sys.stderr,"\nRead from pipe\n\n"
		tkLst = sys.stdin.read().strip().split("\n")
	else:
		tkLst = args
	ticker=tkLst[0]
	nday=7
	if opts['features'] is None:
		feature_list = ['adx','rsi','macd','bband']
	else:
		feature_list = opts['features'].split(',')
	test_size=0.2
	dfa, dfb =run_ML(ticker,nday,feature_list,test_size)
	from sqlalchemy import create_engine, MetaData
	hostname,dbname = (opts['hostname'],opts['dbname'])
	wmode = opts['wmode']
	pgDB = create_engine('postgresql://sfdbo@{}:5432/{}'.format(hostname,dbname))
	tabla = opts['tablename']
	dfa.to_sql(tabla,pgDB,index=False,schema='public',if_exists=wmode)
	tablb = tabla+'_support'
	dfb.to_sql(tablb,pgDB,index=False,schema='public',if_exists=wmode)
