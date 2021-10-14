#!/usr/bin/env python3
'''
Get daily/minute data via yahoo version 8/11 api
Usage of:
yh_hist_batch.py IBM --output=csv --range=1d
OR (To apply 'yh_hchg' as function argument )
yh_hist_batch.py ^GSPC --output=csv --range=5d --gap=1d --extra_xs='funcArg="yh_hchg"
OR
python3 -c "from yh_hist_batch import yh_hist; dx=yh_hist(ticker='^GSPC',gap='1m',ranged='5m');print(dx.to_json(orient='records'))"
OR
python3 -c "from yh_hist_batch import yh_hchg; dx=yh_hchg(ticker='^GSPC',gap='1m',ranged='5m');print(dx.to_json(orient='records'))"

Ref site:
# 1-minute data for 1-day (use indicators=close for selecting columns like IEX filter)
# with validRanges:["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
https://query1.finance.yahoo.com/v8/finance/chart/AAPL?region=US&lang=en-US&includePrePost=false&interval=1m&range=1d

# 1-minute data for 1-day (NOTE, period1,period2 need to be a valid data date due to yahoo's setup) 
# with [start,end] periods:[period1,period2] in epochTime
https://query1.finance.yahoo.com/v8/finance/chart/AAPL?region=US&lang=en-US&includePrePost=false&interval=1m&period1=1553832000&period2=1553918400

# daily data for 1-day
# with validRanges:["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
https://query1.finance.yahoo.com/v8/finance/chart/AAPL?region=US&lang=en-US&includePrePost=false&interval=1d

# multi-ticker 5-minute data for 1-day (use indicators=close for selecting columns like IEX filter)
https://query1.finance.yahoo.com/v7/finance/spark?symbols=AAPL,IBM&range=1d&interval=5m&indicators=close&includeTimestamps=false&includePrePost=false&corsDomain=finance.yahoo.com&.tsrc=finance

# multi-ticker data onTheFly quote 
https://query1.finance.yahoo.com/v7/finance/quote?symbols=AAPL,IBM
https://query1.finance.yahoo.com/v7/finance/quote?fields=symbol,longName,regularMarketPrice,&formatted=false&symbols=MMM,JNJ

# options data (for ticker: amd)
https://query1.finance.yahoo.com/v7/finance/options/amd

# historical daily data
# with [start,end] periods:[period1,period2] in epochTime
https://query1.finance.yahoo.com/v8/finance/chart/AAPL?region=US&lang=en-US&includePrePost=false&interval=1d&period1=1551416400&period2=1554350400

# financial data
https://query2.finance.yahoo.com/v11/finance/quoteSummary/AAPL?modules=summaryProfile,financialData,defaultKeyStatistics
# where module:
assetProfile,balanceSheetHistory,balanceSheetHistoryQuarterly,calendarEvents,cashflowStatementHistory,cashflowStatementHistoryQuarterly,defaultKeyStatistics,earnings,earningsHistory,earningsTrend,financialData,fundOwnership,incomeStatementHistory,incomeStatementHistoryQuarterly,indexTrend,industryTrend,insiderHolders,insiderTransactions,institutionOwnership,majorDirectHolders,majorHoldersBreakdown,netSharePurchaseActivity,price,quoteType,recommendationTrend,secFilings,sectorTrend,summaryDetail,summaryProfile,symbol,upgradeDowngradeHistory,fundProfile,topHoldings,fundPerformance

Last Mod., Wed Apr  3 17:42:40 EDT 2019
'''

#'^GSPC','^TWII','000001.SS','^SOX','^DJI'
import sys
from optparse import OptionParser
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import pytz
import requests
from _alan_calc import pqint,getKeyVal,subDict,subDF

def epoch_parser(x,s=1000): return datetime.fromtimestamp(int(x/s))
def ymd_parser(x,fmt='%Y%m%d'): return datetime.strptime(str(x),fmt)

def eten_hist(ticker=None,gap='1m',ranged='1d',hdrTF=True,tsTF=True,debugTF=False):
	if gap == '1d':
		from _alan_calc import sqlQuery
		xqr="SELECT * FROM prc_hist WHERE name='{}' AND pbdate>20171101".format(ticker)
		return sqlQuery(xqr,dbname='ara.tw')
	twtime = datetime.now(pytz.timezone('Asia/Taipei'))
	if twtime.hour < 9:
		pbdate = (twtime - timedelta(days=1)).strftime('%Y%m%d')
	else:
		pbdate = (twtime).strftime('%Y%m%d')
	return eten_minute(ticker=ticker,pbdate=pbdate,hdrTF=hdrTF,tsTF=tsTF,debugTF=debugTF)

def eten_minute(ticker=None,pbdate=20181120,hdrTF=True,tsTF=True,debugTF=False):
	"""
	Get daily/minute data via eten api
	"""
	if ticker is None:
		return ''
	urx="http://mx68t.etencorp.com:8080/EtenDS/process.php?version=1&objtype=5&extcode={}"
	url=urx.format(ticker)
	if debugTF is True:
		pqint( url, file=sys.stderr)
	try:
		jTmp = pd.read_json(url)['objectData'][0]
	except Exception as e:
		pqint( str(e), file=sys.stderr)
		return {}
	
	#- ARRANGE input data	
	# build output data in datafame
	dx=pd.DataFrame(jTmp['data'])
	dx.loc[:,'ticker']=jTmp['extcode']
	# build date/time as pbdate/epochs column into datafame
	pbdatetime = [datetime.strptime(str(pbdate)+x,'%Y%m%d%H:%M') for x in dx['time'].values]
	dx['epochs'] = [int(x.strftime('%s000')) for x in pbdatetime]
	dx[['open', 'high', 'low', 'close', 'vol']]=dx[['open', 'high', 'low', 'close', 'vol']].astype('float')
	# add datetime index to datafame
	if tsTF is True:
		dx.set_index(pd.DatetimeIndex(pbdatetime),inplace=True)
		dx.index.rename('date',inplace=True)
	# remove NA rows related to [close] data
	dx.rename(columns={'vol':'volume'},inplace=True)
	dx.dropna(subset=['close'],inplace=True)
	# change to prc_temp columns setup
	return dx[['open','high','low','close','volume','epochs','ticker']]

def str2epoch(s,endOfDay=False):
	import re
	if isinstance(s,float):
		s = str(int(s))
	elif isinstance(s,(int,np.integer)):
		s = str(s)
	e = 0
	if len(s)>=10 and s.isdigit():
		e = s[:10]
	elif len(s) in [8,10]:
		s = re.sub('[-/]','',s)
		if len(s)==8 and s.isdigit():
			d = datetime.strptime(s,"%Y%m%d")
			if endOfDay is True:
				d  = d + timedelta(days=1) -  timedelta(seconds=1)
			e = d.strftime("%s")
	return int(e)
	
def yh_rawdata(ticker=None,gap='1m',ranged='1d',debugTF=False):
	"""
	Get daily/minute data via yahoo version 8 api
	"""
	if ticker is None:
		return ''
	if ',' in ranged:
		start, end = ranged.split(',')
		if all([start,end]):
			period1 = str2epoch(start)
			period2 = str2epoch(end,endOfDay=True)
			range_periods = "period1={}&period2={}".format(period1,period2)
		elif  start:
			period1 = str2epoch(start)
			if gap=='1d':
				end = datetime.now().strftime("%Y%m%d")
				period2 = str2epoch(end,endOfDay=True)
			else:
				period2 = (datetime.fromtimestamp(period1)+timedelta(days=1)).strftime("%s")
			range_periods = "period1={}&period2={}".format(period1,period2)
		elif  end:
			period2 = str2epoch(end,endOfDay=True)
			period1 = (datetime.fromtimestamp(period2)+timedelta(days=-1)).strftime("%s")
			#calc beginning time of period1 date
			#if gap=='1d':
			#	d = datetime.fromtimestamp(float(period1))
			#	period1 = datetime.strptime(d.strftime('%Y%m%d'),'%Y%m%d').strftime("%s")
			range_periods = "period1={}&period2={}".format(period1,period2)
		else:
			range_periods = "range=1d"
		urx="https://query1.finance.yahoo.com/v8/finance/chart/{}?region=US&lang=en-US&includePrePost=false&interval={}&{}"
		url=urx.format(ticker,gap,range_periods)
	else:
		urx="https://query1.finance.yahoo.com/v8/finance/chart/{}?region=US&lang=en-US&includePrePost=false&interval={}&range={}"
		url=urx.format(ticker,gap,ranged)
	if debugTF is True:
		pqint( url, file=sys.stderr)
	jX={}
	headers={'Content-Type': 'text/html', 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
	try:
		#jX = pd.read_json(url)
		ret=requests.Session().get(url,headers=headers)
		#ret=requests.get(url,timeout=5)
		jX=ret.json()
	except Exception as e:
		pqint( str(e), file=sys.stderr)
		return {}
	if 'chart' in jX:
		try:
			jTmp = jX['chart']['result'][0]
		except Exception as e:
			return {}
	return jTmp
	
def yh_data(ticker=None,gap='1m',ranged='1d',hdrTF=True,tsTF=True,pchgTF=False,debugTF=False):
	#- PULL raw data	
	jTmp = yh_rawdata(ticker=ticker,gap=gap,ranged=ranged,debugTF=debugTF)

	#- ARRANGE input data	
	# get timestamp
	if 'indicators' in jTmp and 'adjclose' in jTmp['indicators']:
		adjclose = np.array(jTmp['indicators']['adjclose'][0]['adjclose'])
	if debugTF:
		sys.stderr.write("===== Pulling symbol={}&interval={}&range={}...\n".format(ticker,gap,ranged))
	if 'timestamp' not in jTmp:
		sys.stderr.write("**ERROR pulling symbol={}&interval={}&range={}...\n".format(ticker,gap,ranged))
		pqint( jTmp, file=sys.stderr)
		return {}, {}
	pbtimestamp = np.array(jTmp['timestamp'])
	pbdatetime = [epoch_parser(x,1) for x in pbtimestamp]
	# build output data in datafame
	dx=pd.DataFrame(jTmp['indicators']['quote'][0])
	dx.loc[:,'ticker']=ticker
	# build date/time as pbdate/epochs column into datafame
	if gap[-1]=='d': # for daily data
		if 'adjclose' in locals():
			dx['adjusted'] = adjclose
		dx['pbdate'] = [int(x.strftime("%Y%m%d")) for x in pbdatetime]
	else: # for minute data
		dx['epochs'] = pbtimestamp * 1000
	# add datetime index to datafame
	if tsTF is True:
		dx.set_index(pd.DatetimeIndex(pbdatetime),inplace=True)
		dx.index.rename('date',inplace=True)
	# remove NA rows related to [close] data
	dx.dropna(subset=['close'],inplace=True)
	if pchgTF:
		dx=pchg_calc(dx,jTmp,pchgTF)
	return dx, jTmp

def pchg_calc(dx={},jTmp={},pchgTF=False):
	'''
	add change percent
	'''
	if len(dx)<1:
		return {}
	xClose=xChartClose=None
	if 'meta' in jTmp:
		jm = jTmp['meta']
		if 'chartPreviousClose' in jm and jm['dataGranularity'][-1]!='m':
			xChartClose = jm['chartPreviousClose']
		elif 'previousClose' in jm and jm['dataGranularity'][-1]=='m':
			xClose = jm['previousClose']
	if xClose is not None:
		dx['change'] = dx['close']-xClose
		dx['pchg'] = dx['close']/xClose-1
		dx['xclose'] = xClose
	elif xChartClose is not None and pchgTF==2:
		dx['change'] = dx['close']-xChartClose
		dx['pchg'] = dx['close']/xChartClose-1
		dx['xclose'] = xChartClose
	elif xChartClose is not None:
		dx['xclose'] = dx['close'].shift()
		dx['change'] = dx['close'].diff()
		dx['pchg'] = dx['close'].pct_change()
		dx.loc[dx.index[0],'pchg'] = dx.loc[dx.index[0],'close']/xChartClose - 1
		dx.loc[dx.index[0],'change'] = dx.loc[dx.index[0],'close']-xChartClose
		dx.loc[dx.index[0],'xclose'] = xChartClose
	return dx

def yh_hchg(ticker=None,gap='1m',ranged='1d',hdrTF=True,tsTF=True,debugTF=False,**optx):
	optx.update(pchgTF=True)
	dx, jTmp = yh_data(ticker=ticker,gap=gap,ranged=ranged,hdrTF=hdrTF,tsTF=tsTF,debugTF=debugTF,**optx)
	return dx

def yh_hist(ticker=None,gap='1m',ranged='1d',hdrTF=True,tsTF=True,debugTF=False,**optx):
	pchgTF = getKeyVal(optx,'pchgTF',False)
	dx, jTmp = yh_data(ticker=ticker,gap=gap,ranged=ranged,hdrTF=hdrTF,tsTF=tsTF,debugTF=debugTF,**optx)
	# change to prc_temp columns setup
	clx = dx.columns
	if gap[-1]!='m': # for daily data
		if 'ticker' in clx : # for daily data
			dx.rename(columns={'ticker':'name'},inplace=True)
		else:
			dx['name']=ticker
		clx=dx.columns
		colv=['open','high','low','close','volume','adjusted','pbdate','name']
	else:
		colv=['open','high','low','close','volume','epochs','ticker']
	if pchgTF:
		colv += ['xclose','change','pchg']
	colx = [x for x in colv if x in clx ]
	sys.stderr.write("{}\n{}\n{}\n".format(colx,colv,clx))
	sys.stderr.write("{}\n".format(dx[colx].tail()))
	return dx[colx]

def batch_yh_hist(tkLst=[],opts=None,**optx):
	#- Set input parameters
	if opts is None or len(opts)<1:
		opts, _ = opt_yh_hist([])
	if optx is not None:
		opts.update(optx)
	kys=['gap','ranged','tsTF','pchgTF','debugTF']

	debugTF = getKeyVal(opts,'debugTF',False)
	indexTF = getKeyVal(opts,'indexTF',True)
	output = getKeyVal(opts,'output',None)
	sep = getKeyVal(opts,'sep','|')
	#for ky,va in opts.items():
	#	exec('{}=va'.format(ky))

	hdrTF = True
	if 'funcArg' in opts and opts['funcArg'] in globals():
		funcArg = globals()[opts['funcArg']]
	else:
		funcArg = yh_hist
	if len(tkLst)>0 and tkLst[0]=='-':
		tkLst = sys.stdin.read().split("\n")
	dm=pd.DataFrame()
	for j,ticker in enumerate(tkLst):
		hdrTF = True if j<1 else False
		try:
			df = funcArg(ticker,hdrTF=hdrTF,**subDict(opts,kys))
			if len(df)<1:
				continue
			if output is None or len(output)<1:
				dm= pd.concat([dm,df])
		except Exception as e:
			pqint( "**ERROR: {}.{}\n{}".format(j,ticker,str(e)), file=sys.stderr)
			continue
		if output is not None and 'ticker' not in df:
			df['ticker']=ticker
		if output == 'csv':
			sep = sep.encode().decode('unicode_escape') if sys.version_info.major==3 else sep.decode("string_escape")
			sys.stdout.write(df.to_csv(sep=sep,index=indexTF,header=hdrTF) )
		elif output == 'html':
			sys.stdout.write(df.to_html(index=indexTF) )
		elif output == 'json':
			sys.stdout.write(df.to_json(orient='records') )
		hdrTF = False
	return dm

def opt_yh_hist(argv,retParser=False):
	""" command-line options initial setup
	    Arguments:
		argv:   list arguments, usually passed from sys.argv
		retParser:      OptionParser class return flag, default to False
	    Return: (options, args) tuple if retParser is False else OptionParser class
	"""
	parser = OptionParser(usage="usage: %prog [option] SYMBOL1 ...", version="%prog 0.01",
		description="Pull Price History from YAHOO")
	parser.add_option("","--range",action="store",dest="ranged",default='1d',
		help="range period from now (default: 1d)")
	parser.add_option("","--gap",action="store",dest="gap",default='1m',
		help="interval GAP of data frequency (default: 1m)")
	parser.add_option("-d","--database",action="store",dest="dbname",default="ara",
		help="database (default: ara)")
	parser.add_option("","--host",action="store",dest="hostname",default="localhost",
		help="db host (default: localhost)")
	parser.add_option("-t","--table",action="store",dest="tablename",default="prc_temp_yh",
		help="db tablename (default: prc_temp_yh)")
	parser.add_option("-w","--wmode",action="store",dest="wmode",default="replace",
		help="db table write-mode [replace|append] (default: replace)")
	parser.add_option("-o","--output",action="store",dest="output",
		help="OUTPUT type [csv|html|json] (default: no output)")
	parser.add_option("","--no_datetimeindex",action="store_false",dest="tsTF",default=True,
		help="no datetime index (default: use datetime)")
	parser.add_option("","--show_index",action="store_true",dest="indexTF",default=False,
		help="show index (default: False) Note, OUTPUT ONLY")
	parser.add_option("-s","--sep",action="store",dest="sep",default="|",
		help="output field separator (default: |) Note, OUTPUT ONLY")
	parser.add_option("","--no_database_save",action="store_false",dest="saveDB",default=True,
		help="no save to database (default: save to database)")
	parser.add_option("","--extra_js",action="store",dest="extraJS",
		help="extra JSON in DICT format.")
	parser.add_option("","--extra_qs",action="store",dest="extraQS",
		help="extra GET string format like k1=v1&k2=v2; ")
	parser.add_option("","--extra_xs",action="store",dest="extraXS",
		help="extra excutable string like k1=v1;k2=v2; ")
	parser.add_option("","--debug",action="store_true",dest="debugTF",default=False,
		help="debugging (default: False)")
	(options, args) = parser.parse_args(argv[1:])
	try:
		from _alan_str import extra_opts
		opts = vars(options)
		extra_opts(opts,xkey='extraJS',method='JS',updTF=True)
		extra_opts(opts,xkey='extraQS',method='QS',updTF=True)
		extra_opts(opts,xkey='extraXS',method='XS',updTF=True)
		opts.update(args=args,narg=len(args))
	except Exception as e:
		sys.stderr.write(str(e)+"\n")
	if retParser is True:
		return parser
	return (opts, args)

if __name__ == '__main__':
	(opts, tkLst)=opt_yh_hist(sys.argv)
	batch_yh_hist(tkLst,opts)
