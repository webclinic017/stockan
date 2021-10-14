#!/usr/bin/env python3
''' Live datafeed from yahoo quotes
Update tables:
mDB: ara::yh_spark_hist, ara::yh_quote_curr
pgDB: ara::yh_quote_curr (isEoD=True)

Usage of:
+ From commandline: 
python3 yh_chart.py IBM,AAPL
python3 yh_chart.py IBM AAPL
echo "IBM AAPL" | python3 yh_chart.py -

#-------------------------------------------------------------------------
+ For a list of tickers 'tkLst' with direct function call for pricing data
  -- To run Intraday for yh_spark_hist (CRON every 15 minutes 930-1600 M-F)
python3 -c "from yh_chart import yh_spark_hist as ysh;ysh(range='15m',interval='5m',debugTF=True)"
  -- To run EoD yh_quote_curr and yh_spark_hist (CRON at 1600 M-F)
python3 -c "from yh_chart import yh_spark_hist as ysh;ysh(types='quote',isEoD=True,debugTF=True)"
  -- To run SoD yh_quote_curr
python3 -c "from yh_chart import yh_spark_hist as ysh;ysh(types='quote',debugTF=True)"
  -- To run intraday yh_quote_market
printf "^GSPC ^IXIC ^DJI ^SOX" | python3 -c "import sys;from yh_chart import yh_spark_hist as ysh;ysh(sys.stdin.read().split(),t2='market_indicator_quote',types='quote',debugTF=True)"

#-------------------------------------------------------------------------
+ For specific 'ticker' of either non-existed ticker or if not updated in the last 'deltaTolerance' seconds
  -- To setup onTheFly stock quote for non-existed ticker or if not updated in the last 'deltaTolerance' seconds
python3 -c "ticker='AAPL';from yh_chart import yh_quote_comparison as yqc, runOTF;ret=runOTF(yqc,ticker,deltaTolerance=3600,tablename='yh_quote_curr',zpk=['ticker']);"
  -- To setup onTheFly stock intra-day spark history for non-existed ticker or if not updated in the last 'deltaTolerance' seconds
python3 -c "tkLst='AAPL';from yh_chart import yh_spark_hist as ysh, runOTF;ret=runOTF(ysh,ticker,deltaTolerance=900,tablename='yh_spark_hist',zpk=['ticker','pbdt'],range='15m',interval='5m',debugTF=True,dbname='test');"
  -- To setup onTheFly financials 'modules' for non-existed ticker or if not updated in the last 'deltaTolerance' seconds (86400 for 1-day)
python3 -c "ticker='AAPL';from yh_chart import runOTF, yh_financials as yh;ret=runOTF(yh,ticker,deltaTolerance=86400,modules='incomeStatementHistoryQuarterly',tablename='yh_financials',dbname='yh',zpkChk=['ticker','module'],zpk=['ticker','module','pbdate']);"
  -- To setup onTheFly stock intra-day chart history for non-existed ticker or if not updated in the last 'deltaTolerance' seconds (TBD)
python3 -c "tkLst='AAPL';from yh_chart import yh_spark_hist as ysh, runOTF;ret=runOTF(ysh,ticker,deltaTolerance=900,types='chart',tablename='yh_chart_hist',zpk=['ticker','pbdt'],range='15m',interval='5m',debugTF=True,dbname='ara');"

#-------------------------------------------------------------------------
+ From Direct function call for financial keyStatistics
python3 -c "from yh_chart import qS_keyStatistics as yks;yks(saveDB=True,debugTF=True)"
  -- To run Quarterly income_statement
python3 -c "from yh_chart import yh_financials_batch as yf;yf(modules='incomeStatementHistoryQuarterly',tablename='yh_IS_Q',zpk={'ticker','pbdate'});"
python3 -c "from yh_chart import yh_financials_batch as yf;yf(modules='incomeStatementHistoryQuarterly',tablename='yh_IS_Q',zpk={'ticker','pbdate'},useDB=True);"
  -- To run Annual income_statement
python3 -c "from yh_chart import yh_financials_batch as yf;yf(modules='incomeStatementHistory',tablename='yh_IS_A',zpk={'ticker','pbdate'});"
+ NOTE:
  -- below 3 calls are equivalent (1st one perferred, see onTheFlyDB_tst for better reference)
python3 -c "ticker='DAL';from yh_chart import runOTF, yh_financials as yh;ret=runOTF(yh,ticker,deltaTolerance=86400,modules='incomeStatementHistoryQuarterly',tablename='yh_financials',dbname='yh',zpkChk=['ticker','module'],zpk=['ticker','module','pbdate']);"
python3 -c "tkLst=['DAL'];from yh_chart import func2mdb as yhb;dd=yhb(tkLst,dbname='test',tablename='test1',funcN='yh_batchTypes',zpk={'ticker','pbdate'},types='quoteSummary',modules='incomeStatementHistoryQuarterly',debugTF=True);print(dd)"
python3 -c "tkLst=['DAL'];from yh_chart import yh_financials_batch as yf;yf(tkLst,modules='incomeStatementHistory',dbname='test',tablename='test1',zpk={'ticker','module','pbdate'});"
python3 -c "ticker=['AAPL','DAL'];from yh_chart import runOTF, yh_financials as yf;ret=runOTF(yf,ticker,deltaTolerance=86400*90,modules='"summaryProfile"',tablename='"yh_summaryProfile"',dbname='ara',zpkChk=['ticker','module'],zpk=['ticker','module','pbdate']);"

  -- To run summaryProfile  and save to MDB: yh_summaryProfile table
python3 -c "tkLst=['SHAK'];from yh_chart import func2mdb as yhb;dd=yhb(tkLst,tablename='yh_summaryProfile',funcN='yh_batchTypes',zpk=['ticker'],types='quoteSummary',modules='summaryProfile',debugTF=True);print(dd)"

  -- To run summaryProfile and show output
python3 -c "tkLst=['SHAK'];from yh_chart import yh_batchTypes as yhb;dd=yhb(tkLst,types='quoteSummary',modules='summaryProfile',debugTF=True);print(dd)"
  -- To run defaultKeyStatistics and show output
python3 -c "tkLst=['AAPL','IBM'];from yh_chart import yh_batchTypes as yhb;dd=yhb(tkLst,types='quoteSummary',modules='defaultKeyStatistics',debugTF=True);print(dd)"
  -- To add quote of tickers
python3 -c "from yh_chart import func2mdb as yhq; yhq(['XLRE'],tablename='yh_quote_curr',dbname='ara',funcN='yh_quote_comparison',zpk={'ticker'});"

  -- DEPRECATED
python3 -c "from yh_chart import yh_quote_curr as yqc;d=yqc()"
python3 -c "from yh_chart import func2mdb as yhq; yhq(['XLRE'],tablename='yh_quote_curr',dbname='ara',funcN='yh_quote_comparison',zpk={'ticker'});"
python3 -c "from yh_chart import yh_hist_query as yhq;d=yhq(['AAPL'],types='spark');print(d)"
python3 -c "from yh_chart import yh_hist_query as yhq;d=yhq(['AAPL'],types='quote');print(d)"
python3 -c "from yh_chart import yh_quote_comparison as yqc;print(yqc(['AAPL']))"

Last mod.,
Wed May 27 10:20:54 EDT 2020
-----------------------------------------------------------------------------------------------------
'''
import sys, datetime
import requests
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from _alan_str import write2mdb,find_mdb,insert_mdb,upsert_mdb,get_arg2func
from _alan_calc import getKeyVal,conn2pgdb,conn2mgdb,renameDict,subDict,subDF,saferun,safeRunArg
headers={'Content-Type': 'text/html', 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}

def epoch_parser(x,s=1000): return datetime.datetime.fromtimestamp(int(x/s))

def pull_act_tickers(sql=None,engine=None):
	if not sql:
		sql="select ticker from mapping_ticker_cik where act_code=1 ORDER BY ticker"
	try:
		from _alan_calc import sqlQuery
		df = sqlQuery(sql,engine=engine)
	except Exception as e:
		df={}
	return df

def list2chunk(v,n=100):
	''' partition an array 'v' into arrays limit to 'n' elements
	'''
	import numpy as np
	return [v[i:i+max(1,n)] for i in np.arange(0, len(v), n)]


def yh_spark_hist(tkLst=None,**optx):
	dd = yh_batchSpark(tkLst=tkLst,**optx)
	if optx.pop('dfTF',False) and not isinstance(dd,pd.DataFrame):
		dd = pd.DataFrame(dd)
	if optx.pop('tsTF',False) and isinstance(dd,pd.DataFrame) and \
		not isinstance(dd.index,(datetime.date, datetime.datetime)):
		if 'pbdt' in dd.columns:
			dd.set_index('pbdt',drop=True,inplace=True)
		elif 'epochs' in dd.columns:
			pbdt = np.vectorize(epoch_parser)(dd['epochs'])
			dd.set_index(pd.DatetimeIndex(pbdt),inplace=True)
		dd.index.name='date'
	return dd

def useWeb(jobj={'ticker':'^GSPC'},colx='pbdt',dbname='ara',tablename='yh_quote_curr',mmGap=30,**optx):
	'''return [True|Flase] action for using web or DB
	based on 'tablename' 'colx' field

	'''
	from _alan_calc import conn2mgdb
	debugTF=getKeyVal(optx,'debugTF',False)
	webTF=optx.pop('webTF',None)
	if webTF is not None and isinstance(webTF,bool):
		return webTF
	cdt=getKeyVal(optx,'cdt',datetime.datetime.now())
	mgDB=conn2mgdb(dbname=dbname)

	dc = mgDB[tablename].find_one({"$query":jobj,"$orderby":{colx:-1}},{colx:1,"_id":0})
	if len(dc)<1:
		return True
	pdt=dc[colx]
	mmPassed=pd.Timedelta(cdt - pdt).total_seconds() / 60.0
	webTF = mmPassed>mmGap
	if debugTF is True:
		sys.stderr.write("{}|{}|{}|{}|{}\n".format("webTF","cdt","pdt","mmPassed","mmGap"))
		sys.stderr.write("{}|{}|{}|{:.0f}|{}\n".format(webTF,cdt,pdt,mmPassed,mmGap))
	return webTF

def yh_quote_curr(tkLst=None,screenerTF=False,dbname='ara',tablename='yh_quote_curr',zpk={'ticker'},m2pTF=False,webTF=None,**optx):
	'''
	create yh_quote_curr table
	'''
	if tkLst is None or len(tkLst)<1:
		from _alan_calc import sqlQuery
		tkDF=sqlQuery('select ticker from mapping_ticker_cik where act_code=1')
		tkLst=list(tkDF['ticker'])
	try:
		jobj=dict(ticker=tkLst[0])
		if webTF is None:
			webTF = useWeb(jobj=jobj,dbname=dbname,tablename=tablename,**optx)
		if not webTF:
			jobj = {'ticker':{'$in':tkLst}}
			d,_,_ = find_mdb(jobj,dbname=dbname,tablename=tablename,dfTF=True)
			sys.stderr.write("===Using data from {}::{}\n".format(dbname,tablename))
			return d
		d=yh_quote_comparison(tkLst,screenerTF=screenerTF,dbname=dbname,tablename=tablename,zpk=zpk,**optx)
	except Exception as e:
		sys.stderr.write("**ERROR: {}:{}:{}\n".format("yh_quote_curr()","@ yh_quote_comparison",str(e)))
		d=[]
	saveDB=optx.pop('saveDB',False)
	if not saveDB:
		return d
	try:
		if m2pTF:
			from mongo2psql import mongo2psql
			mongo2psql(tablename,dbname)
			sys.stderr.write("===Running mongo2psql()...{dbname}::{tablename}".format(**locals()))
	except Exception as e:
		sys.stderr.write("**ERROR: {}:{}:{}\n".format("yh_quote_curr()","mongo2psql ...",str(e)))

	# Save yh_quote_curr To yh_quote_hist
	try:
		dbM = conn2mgdb(dbname=dbname)
		tablename = tablename.replace('curr','hist')
		ret=dbM[tablename].insert_many(d.to_dict(orient='records'),ordered=False)
		sys.stderr.write("===Saving history: {dbname}::{tablename}".format(**locals()))
	except Exception as e:
		sys.stderr.write("**ERROR: {}:{}:{}\n".format("yh_quote_curr()","MDB saving...",str(e)))
	return d

def qS_keyStatistics(tkLst=[],tablename='qS_keyStatistics',saveDB=False,debugTF=True,**optx):
	''' pull defaultKeyStatistics from yahoo and save to qS_keyStatistics
	e.g.,
	python3 -c "from yh_chart import qS_keyStatistics as yks;yks(['AAPL']);"
	
	'''
	if tkLst is None or len(tkLst)<1:
		tkLst = list(pull_act_tickers()['ticker'])
	dbM=None
	pbdt=datetime.datetime.now()
	jdM=yh_batchTypes(tkLst,types='quoteSummary',modules='defaultKeyStatistics',debugTF=debugTF)
	for jx in jdM:
		jx['pbdt']=pbdt
	if saveDB is True:
		zpk=optx.pop('zpk',{"ticker"})
		m,dbM,err=upsert_mdb(jdM,clientM=dbM,tablename=tablename,zpk=zpk,**optx)
		sys.stderr.write("=== {} of {} saved to {}\n".format(tkLst,jdM[-1],tablename))
	return jdM

def yh_financials_batch(tkLst=[],**optx):
	''' pull financials from yahoo and save to 'tablename'
	e.g.,
	python3 -c "from yh_chart import yh_financials_batch as yf;yf(['AAPL']);"
	OR
	python3 -c "from yh_chart import yh_financials_batch as yf;yf(modules='incomeStatementHistoryQuarterly',tablename='yh_IS_Q',zpk={'ticker','pbdate'});"
	'''
	useDB=optx.pop('useDB',False)
	tablename=optx.pop('tablename','')
	modules=optx.pop('modules','incomeStatementHistory')
	if len(tablename)>0 or useDB is True:
		dbname=optx.pop('dbname','ara')
		dbM=conn2mgdb(dbname=dbname)
		if useDB is True and (tkLst is None or len(tkLst)<1):
			tkLst=dbM['yh+'+modules].distinct("ticker")
	else:
		dbM=None
	if tkLst is None or len(tkLst)<1:
		tkLst = list(pull_act_tickers()['ticker'])
	sys.stderr.write("===Batch list:{}, tablename: {}, useDB:{}\n".format(tkLst,tablename,useDB))
	dd=[]
	for tk in tkLst:
		try:
			jd = yh_financials(tk,modules=modules,tablename=tablename,clientM=dbM,useDB=useDB,**optx)
		except Exception as e:
			sys.stderr.write("**ERROR:{}:{} @{}\n".format(tk,str(e),'yh_financials'))
			continue
		dd.extend(jd)
	return dd

def epoch2ymd(eptime):
	return int(datetime.datetime.fromtimestamp(eptime).strftime('%Y%m%d'))

# individual run of yh_financials_batch
def yh_financials(ticker,modules='incomeStatementHistory',saveDB=False,clientM=None,useDB=False,dbname='ara',tablename='',**optx):
	''' pull financials from yahoo and save to 'tablename'
	e.g.,
	python3 -c "from yh_chart import yh_financials as yf;yf('AAPL');"
	
	'''
	debugTF=getKeyVal(optx,'debugTF',False)
	types=getKeyVal(optx,'types','quoteSummary')
	xmod=modules
	jdM=[]
	if useDB is True:
		jdTmp,dbM,err=find_mdb({"ticker":ticker},clientM,dbname=dbname,tablename='yh_'+xmod)
		if len(jdTmp)<1:
			return []
		jdMod = jdTmp[0]
	else:
		try:
			jdTmp=yh_quoteSummary(ticker,modules=xmod,**optx)
			jdMod = jdTmp['quoteSummary']['result'][0][xmod]
			jD=proc_summaryQuote(jdMod,ticker=ticker,xmod=xmod)
			if jD:
				jdM=[jD] if not isinstance(jD,list) else jD
		except:
			sys.stderr.write("**ERROR:{} not found via {}\n".format(ticker,modules))
	if saveDB is True and len(tablename)>0 and len(jdM)>0:
		m,dbM,err=insert_mdb(jdM,clientM=clientM,tablename=tablename,**optx)
		sys.stderr.write("=== {} of {} saved to {}\n".format(ticker,jdM[-1],tablename))
		sys.stderr.write("{}\n".format(pd.DataFrame(jdM)[['ticker','module']].to_string(index=False)))
	return jdM

def yh_quote_comparison(tkLst=None,types='quote',sortLst=['changePercent'],rawTF=True,screenerTF=False,dfTF=False,debugTF=False,ascendingTF=False,**optx):
	'''
	show a list of ticker change / volume
	default for sector/industry performance ranking like screener
	'''
	if tkLst is None or len(tkLst)<1:
		tkLst = ['XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','XTL']
	elif isinstance(tkLst,str):
		tkLst = [tkLst]
	if debugTF:
		sys.stderr.write("===yh_quote_comparison() LOCALS:\n{}".format(locals()))
	df=yh_hist_query(tkLst,types=types,rawTF=rawTF,screenerTF=screenerTF,debugTF=debugTF,dfTF=dfTF,**optx)
	if not dfTF:
		return df
	df = pd.DataFrame(data=df)
	newNames={"symbol":"ticker","regularMarketPrice":"close","regularMarketChange":"change","regularMarketChangePercent":"changePercent","regularMarketOpen":"open","regularMarketDayHigh":"high","regularMarketDayLow":"low","regularMarketVolume":"volume"}
	df.rename(newNames,axis='columns',inplace=True)
	if sortLst and len(sortLst)>0 and sortLst[0] in df:
		df = df.sort_values(by=sortLst,ascending=ascendingTF)
		df.reset_index(drop=True,inplace=True)
	return df

def yh_quoteSummary(ticker,modules='summaryProfile',debugTF=False,**optx):
	if modules=='*':
		modules="""assetProfile,balanceSheetHistory,balanceSheetHistoryQuarterly,calendarEvents,cashflowStatementHistory,cashflowStatementHistoryQuarterly,defaultKeyStatistics,earnings,earningsHistory,earningsTrend,financialData,fundOwnership,incomeStatementHistory,incomeStatementHistoryQuarterly,indexTrend,industryTrend,insiderHolders,insiderTransactions,institutionOwnership,majorDirectHolders,majorHoldersBreakdown,netSharePurchaseActivity,price,quoteType,recommendationTrend,secFilings,sectorTrend,summaryDetail,summaryProfile,symbol,upgradeDowngradeHistory,fundProfile,topHoldings,fundPerformance"""
	urx = 'https://query2.finance.yahoo.com/v11/finance/quoteSummary/{ticker}?modules={modules}'
	url=urx.format(ticker=ticker,modules=modules)
	if debugTF is True:
		sys.stderr.write("==URL:\n{}\n".format(url))
	ret = requests.Session().get(url,headers=headers)
	#ret = requests.get(url,timeout=3)
	jdTmp = ret.json()
	return jdTmp

def yh_batchRaw(tkLst=[],filter='*',types='spark',range='5m',interval='1m',debugTF=False,**optx):
	'''
	Pulling finance.yahoo data based on the 'types' of [spark|quote|chart]
	'''
	if tkLst is None or len(tkLst)<1:
		return {}
	if types in ['spark','quote']:
		utmp = 'https://query1.finance.yahoo.com/v7/finance/{}?corsDomain=finance.yahoo.com&.tsrc=finance&'
		urx = utmp + 'symbols={}&range={}&interval={}'
		url = urx.format(types,','.join(tkLst),range,interval)
	else:
		utmp = 'https://query1.finance.yahoo.com/v8/finance/{}/{}?corsDomain=finance.yahoo.com&.tsrc=finance&'
		urx = utmp + 'range={}&interval={}'
		tkX = tkLst[0] if isinstance(tkLst,(list,tuple)) else tkLst
		url = urx.format(types,tkX,range,interval)
	includeTimestamps = optx['includeTimestamps'] if 'includeTimestamps' in optx else True
	includePrePost = optx['includePrePost'] if 'includePrePost' in optx else False
	url += "&includeTimestamps={}&includePrePost={}".format(includeTimestamps,includePrePost)
	if filter != '*' and types=='quote':
		ftx="&fields={}".format(filter)
		url += ftx
	elif filter != '*':
		ftx="&indicators={}".format(filter)
		url += ftx
	if debugTF is True:
		sys.stderr.write("==URL:\n{}\n".format(url))
	res=requests.Session().get(url,headers=headers)
	#res=requests.get(url,timeout=5)
	if debugTF:
		sys.stderr.write("=====URL STATUS:{}\n{}\n{}\n".format(res.status_code,tkLst,url))
	if res.status_code!=200:
		return {}
	try:
		jdTmp = res.json()
	except Exception as e:
			sys.stderr.write("**ERROR: {}:{}:{}\n".format("yh_batchRaw()",tkLst,str(e)))
			return {}
	return jdTmp

def batchSpark_quoteConv(jdLst=[],debugTF=False,**optx):
	colX=getKeyVal(optx,'colX',[])
	dicX={"symbol":"ticker","regularMarketPrice":"close","regularMarketChange":"change","regularMarketChangePercent":"changePercent","regularMarketOpen":"open","regularMarketDayHigh":"high","regularMarketDayLow":"low","regularMarketVolume":"volume"}
	def quote_dx2dd(jdTmp,dicX,colX):
		if "regularMarketTime" not in jdTmp:
			sys.stderr.write("**WARNING:{} {}\n".format(ticker,"is invalid."))
			return {}
		renameDict(jdTmp,dicX)
		#keep "regularMarketPreviousClose" for backward compatibility
		if "regularMarketPreviousClose" in jdTmp:  
			jdTmp['xclose']=jdTmp["regularMarketPreviousClose"]
		if "changePercent" in jdTmp:
			jdTmp['pchg']=jdTmp["changePercent"]/100.0
		epoch = int(jdTmp["regularMarketTime"])
		jdTmp['epochs']=epoch*1000
		jdTmp['pbdt']=pbdt=datetime.datetime.fromtimestamp(epoch)
		jdTmp['hhmm']=pbdt.strftime('%H%M')
		jdTmp['pbdate']=int(pbdt.strftime('%Y%m%d'))
		dd = subDict(jdTmp,colX) if len(colX)>0 else jdTmp
		return dd

	jdM=[]
	for jdTmp in jdLst:
		ticker = jdTmp['symbol']
		try:
			dd = quote_dx2dd(jdTmp,dicX,colX)
			if len(dd)<1:
				continue
			jdM.append(dd)
		except Exception as e:
			sys.stderr.write("**ERROR:{}, {}\n".format(ticker,str(e)))
			continue
	return jdM

def batchSpark_sparkConv(jdLst=[],debugTF=False,**optx):
	def spark_dx2dd(jdX):
		ticker = jdX['symbol']								
		jdd = jdX['response'][0]
		jde={}
		xClose=xChartClose=0
		if 'timestamp' not in jdd:
			return jde,xClose,xChartClose,''
		if 'meta' in jdd and 'chartPreviousClose' in jdd['meta']:
			xChartClose = jdd['meta']['chartPreviousClose']
		if 'meta' in jdd and 'previousClose' in jdd['meta']:
			xClose = jdd['meta']['previousClose']
		try:
			gap=jdd['meta']['dataGranularity']
			epoch = jdd['timestamp']
			close=jdd['indicators']['quote'][0]['close']
			pbdt = [ datetime.datetime.fromtimestamp(int(x)) for x in epoch ]
			epochs = [ x*1000 for x in epoch ]
			hhmm = [x.strftime('%H%M') for x in pbdt]
			pbdate = [int(x.strftime('%Y%m%d')) for x in pbdt]
			ne = len(epoch)
		except Exception as e:
			sys.stderr.write("**ERROR:{}, {}\n".format(ticker,str(e)))
			return jde, xClose,xChartClose,gap
		jde['ticker']=[ticker]*ne
		jde['close']=close
		jde['epochs']=epochs
		jde['pbdt']=pbdt
		jde['hhmm']=hhmm
		jde['pbdate']=pbdate
		return jde,xClose,xChartClose,gap
	jdM=[]
	for jdX in jdLst:
		jde,xClose,xChartClose,gap = spark_dx2dd(jdX)
		if len(jde)<1:
			continue
		else:
			dx = pd.DataFrame(data=jde)
			dx = dx.dropna()
			# NOTE, 
			# Minute data 'change' is calc based on last-day close instead of 'gap' interval
			# This can be be a potential bug if range is more than 1-d
			if gap[-1] in ['m'] and xClose>0: # for minute data with '1d' range ONLY
				dx['xclose'] = xClose
				dx['change'] = dx['close'] - dx['xclose']
				dx['pchg'] = dx['change']/dx['xclose']
			elif gap[-1] not in ['m'] and xChartClose>0: # for minute data
				dx['xclose'] = dx['close'].shift()
				dx.loc[0,'xclose'] = xChartClose
				dx['change'] = dx['close'] - dx['xclose']
				dx['pchg'] = dx['change']/dx['xclose']
			else:
				dx['xclose'] = dx['pchg'] = dx['change'] = 0.0
			dd = dx.to_dict(orient='records')
			jdM.extend(dd)
	return jdM

def chart2df(jdLst=[],**optx):
	#- get DF
	jdTmp=jdLst[0]
	dx=pd.DataFrame(jdTmp['indicators']['quote'][0])

	# get timestamp
	pbts = np.array(jdTmp['timestamp'])
	pbdt = np.vectorize(datetime.datetime.fromtimestamp)(pbts)
	dx['pbdt'] = pbdt
	dx['pbdate'] = [int(x.strftime("%Y%m%d")) for x in pbdt]
	dx['epochs'] = pbts * 1000

	# get ticker
	dx.loc[:,'ticker']=jdTmp['meta']['symbol']

	# Use previousClose to calc change and pchg (changePercent)
	# NOTE, 
	# Minute data 'change' is calc based on last-day close instead of 'gap' interval
	# This can be be a potential bug if range is more than 1-d
	gap=jdTmp['meta']['dataGranularity']
	if gap[-1]=='m': # for minute data
		dx.loc[:,'xclose']=jdTmp['meta']['previousClose']
	else:
		xclose=jdTmp['meta']['chartPreviousClose']
		dx.loc[:,'xclose']=dx['close'].shift()
		dx.loc[0,'xclose'] = xclose

	# build date/time as pbdate/epochs column into datafame
	if 'indicators' in jdTmp and 'adjclose' in jdTmp['indicators']:
		adjclose = np.array(jdTmp['indicators']['adjclose'][0]['adjclose'])
		dx['adjusted'] = adjclose
	# remove NA rows related to [close] data
	dx.dropna(subset=['close'],inplace=True)
	dx['change']=dx['close']-dx['xclose']
	dx['pchg']=dx['change']/dx['xclose']
	return dx.to_dict(orient='records')

def yh_batchSpark(tkLst=[],filter='*',types='spark',nchunk=100,saveDB=True,dbname='ara',tablename='yh_spark_hist',zpk={'ticker','epochs'},t2='yh_quote_curr',t2zpk={'ticker'},isEoD=False,**optx):
	'''
	pull data from types=[spark|quote] then save to mgDB 'dbname':: 'tablename' and 't2' respectively
	Note,
	if isEoD=True: quote data save to  both mgDB 'dbname':: 'tablename' and 't2'
	if tablename or t2 ='' , nothing will be saved in 'tablename' or 't2'
	quote data will also be saved to pgDB 'dbname'::'t2' if isEoD=True and types='quote'
	'''
	debugTF=getKeyVal(optx,'debugTF',False)
	if debugTF:
		sys.stderr.write("===LOCALS: {}\noptx: {}\n".format(locals(),optx))
	dbM=conn2mgdb(dbname=dbname)
	if tkLst is None or len(tkLst)<1:
		tkLst = list(pull_act_tickers()['ticker'])
	elif isinstance(tkLst,str):
		tkLst = [tkLst]
	chunkLst = list2chunk(tkLst,nchunk)
	jdN=[]
	colX=['ticker','open','high','low','close','xclose','volume','pchg','change','pbdate','epochs','pbdt','hhmm']
	for j,tkM in enumerate(chunkLst):
		jdTmp = yh_batchRaw(tkM,types=types,**optx)
		jdQC = {}

		if types in ['spark','chart']:
			jdLst = jdTmp[types]['result']
			if types=='chart':
				jdM = chart2df(jdLst,**optx)
			else:
				jdM = batchSpark_sparkConv(jdLst,**optx)
			if len(jdM)<1:
				continue
			if saveDB is True:
				m,dbM,err=insert_mdb(jdM,clientM=dbM,tablename=tablename,**optx)
				if debugTF:
					sys.stderr.write("=== {} of {} saved to {}\n".format(tkM,jdM[-1],tablename))
			jdN.extend(jdM)
		else: # for case of types.lower()=='quote'
			jdLst = jdTmp['quoteResponse']['result']
			jdQC = batchSpark_quoteConv(jdLst,**optx)
			if len(jdQC)<1:
				continue
			jdM = subDict(jdQC,colX)
			if saveDB is True and len(jdQC)>0:
				if len(t2)>0:
					for tname in t2.split(','):
						qc,dbM,erq=upsert_mdb(jdQC,clientM=dbM,tablename=tname,zpk=t2zpk,**optx)
						sys.stderr.write("=== {} of {} saved to {}\n".format(tkM,jdQC[-1],tname))
				if isEoD is True:
					m,dbM,err=insert_mdb(jdM,clientM=dbM,tablename=tablename,zpk=zpk,**optx)
					sys.stderr.write("=== {} of {} saved to {}\n".format(tkM,jdM[-1],tablename))
			jdN.extend(jdQC)
		if debugTF:
			sys.stderr.write("=== tkM:{}[{}/{}], last:\n{}\n".format(tkM,len(jdM),len(jdN),jdN[-1]))

	if saveDB is True and len(t2)>0 and len(jdN)>0 and types.lower()=='quote' and isEoD is True:
		try:
			df = pd.DataFrame(jdN)
			df = df.drop(['_id'], axis=1)
			if debugTF:
				sys.stderr.write("=== Save to pgDB::{} of {} rows(1st-last)\n{}\n{}\n".format(t2,len(df),df.iloc[:1],df.iloc[-1:]))
			from _alan_calc import save2pgdb
			for tname in t2.split(','):
				save2pgdb(df,dbname,tablename=tname)
		except Exception as e:
			sys.stderr.write("**ERROR: {}:{}\n".format("save2pgdb",str(e)))
	return jdN

def proc_raw_fmt(dc):
	sys.stderr.write("===proc_raw_fmt {}:{}".format(type(dc),dc)[:100]+"\n")
	if isinstance(dc,list):
		for jk,jdc in enumerate(dc):
			dc[jk]=proc_raw_fmt(jdc)
		return dc
	elif not isinstance(dc,dict):
		return {}
	jdc = {}
	for ky,val in dc.items():
		if ky == 'maxAge':
			continue
		elif not val:
			continue
		cv=val
		if isinstance(val, list):
			cv=proc_raw_fmt(val)
		elif isinstance(val, dict):
			if 'raw' in val:
				cv=val['raw']
		jdc.update({ky:cv}) 
		
	return jdc

def proc_summaryQuote(modObj={},xmod='',ticker=''):
	if not all([modObj,xmod,ticker]):
		return {}
	outLst= proc_raw_fmt(modObj)
	ktmp=list(outLst)
	if len(ktmp)==1 and isinstance(outLst[ktmp[0]],list):
		ymod=ktmp[0]
		outLst=outLst[ymod]
		for j,jx in enumerate(outLst):
			if 'endDate' in outLst[j] and outLst[j]['endDate']>10**9:
				outLst[j]['pbdate']= epoch2ymd(outLst[j]['endDate'])
			elif 'quarter' in outLst[j] and outLst[j]['quarter']>10**9:
				outLst[j]['pbdate']= epoch2ymd(outLst[j]['quarter'])
			elif 'reportDate' in outLst[j] and outLst[j]['reportDate']>10**9:
				outLst[j]['pbdate']= epoch2ymd(outLst[j]['ReportDate'])
			outLst[j]['module']=xmod
			outLst[j]['ticker']=ticker
			if ymod:
				outLst[j]['submodule']=ymod
	else:
		ymod=None
		kyLst=list(outLst.keys())
		for ky in kyLst:
			outLst['module']=xmod
			outLst['ticker']=ticker
	return outLst

def process_summaryRawData(dc={}):
	dd={}
	if not isinstance(dc,dict):
		return {}
	for ky,va in dc.items():
		if isinstance(va,dict):
			if 'raw' in va:
				dd.update({ky:va['raw']})
		else:
			if va is not None:
				dd.update({ky:va})
	if 'endDate' in dd and dd['endDate']>10**9:
		dd['pbdate']= epoch2ymd(dd['endDate'])
	elif 'quarter' in dd and dd['quarter']>10**9:
		dd['pbdate']= epoch2ymd(dd['quarter'])
	return dd

def yh_batchTypes(tkLst=[],filter='*',types='quote',nchunk=100,modules='summaryProfile',debugTF=False,**optx):
	#xlst = (unicode, str) if hasattr(__builtins__,"unicode") else str
	#if isinstance(tkLst,xlst):
	#	chunkLst = [tkLst]
	if types in ['quote','spark']:
		chunkLst = list2chunk(tkLst,nchunk)
	else:
		chunkLst = tkLst
	jdM=[]
	for j,tkM in enumerate(chunkLst):
		try:
			if types=='quoteSummary': # tkM is a string of ticker name
				jdTmp = yh_quoteSummary(tkM,modules=modules,debugTF=debugTF,**optx)
			else: # tkM is a list of tickers
				jdTmp = yh_batchRaw(tkM,filter=filter,types=types,debugTF=debugTF,**optx)
			if not isinstance(jdTmp,dict) or len(jdTmp)<1:
				continue
			if types=='quoteSummary': # tkM is a ticker string
				try:
					jdLst = jdTmp['quoteSummary']['result'][0]
				except:
					continue
				for xmod in jdLst:
					modObj=jdLst[xmod]
					jD=proc_summaryQuote(modObj,ticker=tkM,xmod=xmod)
				jdX=[jD] if not isinstance(jD,list) else jD
			elif types=='quote':
				jdX = jdTmp['quoteResponse']['result']
			else:
				jdX = jdTmp[types]['result']
			if debugTF:
				sys.stderr.write("==={}.{}\njdTmp\n{}\njdX\n{}\n".format(j,tkM,jdTmp,jdX[-1]))
		except Exception as e:
			sys.stderr.write("**ERROR: {}:{}\n".format(types,str(e)))
			continue
		jdM.extend(jdX)
	if debugTF:
		sys.stderr.write("===jdM:\n{}".format(jdM))
	return jdM

# DEPRECATED
def raw2screener_output_1(jobj):
	d = dict(ticker=jobj['symbol'])
	d.update(changePercent=jobj['regularMarketChangePercent'])
	if 'marketCap' in jobj:
		d.update(marketCap=jobj['marketCap'])
	else:
		d.update(marketCap=0)
	for xk,ky in zip(['ytdReturn','regularMarketVolume','regularMarketPrice','regularMarketChange','shortName'],['ytdReturn','volume','price','change','company']) :
		if xk in jobj:
			d.update({ky:jobj[xk]})
	return d

def raw2spark_output(jobj):
	from _alan_calc import renameDict
	colx=['ticker','close','changePercent','regularMarketPreviousClose','epochs','pbdt','change']
	ds={x:y for x,y in jobj.items() if x in colx}
	renameDict(ds,{'regularMarketPreviousClose':'xclose','changePercent':'pchg'})
	ds['hhmm']=ds['pbdt'].strftime('%H%M')
	ds['pchg'] /= 100.0
	return ds

def yh_hist_query(tkLst=[],filter='*',types='quote',nchunk=50,rawTF=False,screenerTF=False,dfTF=False,debugTF=False,dbname=None,tablename=None,**optx):
	'''
	Pull minute ohlc pricing data from Yahoo but use marketVolume as volume
	since market data has 15-minute delay, latest 15 marketVolumes become 0 
	'''
	if len(tkLst)<1:
		tkLst = list(pull_act_tickers()['ticker'])
	jdLst = yh_batchTypes(tkLst,filter=filter,types=types,nchunk=nchunk,debugTF=debugTF,**optx)
	colX = ["ticker","open","high","low","close","volume","xclose","change","pchg","epochs",'hhmm',"pbdate","pbdt"]
	dLst=[]
	df=pd.DataFrame()
	clientM=None
	tablename = 'yh_{}_temp'.format(types) if tablename is None else tablename
	for j,jdTmp in enumerate(jdLst):
		try:
			jdX={}
			if 'response' in jdTmp:
				ticker = jdTmp['symbol']
				jdd = jdTmp['response'][0]
			elif 'meta' in jdTmp:
				ticker = jdTmp['meta']['symbol']
				jdd = jdTmp
			else: #- for 'quote' parsing
				if "regularMarketPrice" not in jdTmp:
					continue
				if "regularMarketTime" not in jdTmp:
					continue
				jdTmp['epochs']=jdTmp['regularMarketTime']*1000
				jdTmp['pbdt'] = datetime.datetime.fromtimestamp(jdTmp['regularMarketTime'])
				jdTmp['pbdate'] = int(jdTmp['pbdt'].strftime('%Y%m%d'))
				newNames={"symbol":"ticker","regularMarketPrice":"close","regularMarketChange":"change","regularMarketChangePercent":"changePercent","regularMarketOpen":"open","regularMarketDayHigh":"high","regularMarketDayLow":"low","regularMarketVolume":"volume","regularMarketPreviousClose":"xclose","regularMarketTime":"epoch"}
				if rawTF:
					renameDict(jdTmp,mapper=newNames)
					if debugTF:
						sys.stderr.write("{}\n".format(jdTmp))
					if screenerTF==True:
						colx=["change","changePercent","company","marketCap","close","ticker","volume","epochs","pbdt"]
						#ds=raw2screener_output_1(jdTmp)
						ds =  subDict(jdTmp,colx)
						renameDict(ds,{"close":"price"})
					elif screenerTF>0: # original False case
						colx=list( set(newNames.values()).union(['epochs','pbdt','hhmm','pbdate','marketCap']) )
						ds =  subDict(jdTmp,colx)
					else:
						ds = jdTmp 
					if all([dbname,tablename]):
						zpk=getKeyVal(optx,'zpk',['ticker','epochs'])
						#mobj,clientM,err_msg = write2mdb(ds,clientM,dbname=dbname,tablename=tablename,zpk=zpk)
						mobj,clientM,err_msg = insert_mdb(ds,clientM,dbname=dbname,tablename=tablename,zpk=zpk)
						if debugTF:
							sys.stderr.write("{}\nSave to {}::{}\n".format(ds,dbname,tablename))
					dLst.append(ds)
					continue
				#- for 'spark' and 'chart' parsing
				dx=pd.DataFrame([jdTmp])
				dx.rename(newNames,axis='columns',inplace=True)
				if 'volume' not in dx:
					continue
				dx.dropna(subset=['volume'],inplace=True)
				if len(dx)<1:
					continue
				colX = [x for x in colX if x in dx.columns]
				dm = dx[colX]
				if debugTF:
					sys.stderr.write("quote:\n{}".format(dm.tail()))
				df= pd.concat([df,dm])
				continue
			xClose=None
			if 'meta' in jdd and 'previousClose' in jdd['meta']:
				xClose = jdd['meta']['previousClose']
			epoch = jdd['timestamp']
			for x,y in jdd['indicators']['quote'][0].items():
				jdX[x] = y
			jdX['epochs'] = np.array(epoch)*1000
			dx=pd.DataFrame(jdX)
			dx['ticker']=ticker
			if 'pchgTF' in optx and optx['pchgTF'] is False:
				df= pd.concat([df,dx])
				continue
			elif 'pchgTF' in optx and optx['pchgTF'] and jdd['meta']['dataGranularity'][:1]!='m':
				dx['pchg'] = dx['close'].pct_change()
				dx['change'] = dx['close'].diff()
				xChartClose = jdd['meta']['chartPreviousClose']
				dx.loc[dx.index[0],'pchg'] = dx.loc[dx.index[0],'close']/xChartClose - 1
				dx.loc[dx.index[0],'change'] = dx.loc[dx.index[0],'close']-xChartClose

			pbdt = [ datetime.datetime.fromtimestamp(int(x)) for x in epoch ]
			dx['hhmm']=[x.strftime('%H%M') for x in pbdt]
			dx['pbdate']=[x.strftime('%Y%m%d') for x in pbdt]
			dx['pbdate']=dx['pbdate'].astype(int)
			dx['pbdt']=pbdt
			dx = dx.dropna()
			if xClose is not None and xClose>0:
				dx['pchg'] = dx['close']/xClose -1
				dx['change'] = dx['close'] - xClose
				dx['xclose'] = xClose
			colX = [x for x in colX if x in dx.columns]
			dm = dx[colX]
			if debugTF:
				sys.stderr.write("{}".format(dm.tail()))
			if dfTF:
				df= pd.concat([df,dm])
			else:
				dLst.extend(dm.to_dict(orient='records'))
		except Exception as e:
			sys.stderr.write("**ERROR: {}:{}:{}\n".format(j,jdTmp,str(e)))
			continue
	if len(df)>0:
		df.reset_index(drop=True,inplace=True)
	if len(dLst)>0:
		if dfTF:
			dLst = pd.DataFrame(dLst)
		return dLst
	return df

def func2mdb(tkLst,tablename='yh_spark_hist',dbname='ara',funcN='yh_hist_query',zpk={'ticker','hhmm'},debugTF=False,**optx):
	'''
	Run 'funcN'() and save the result to mongoDB
	'''
	from _alan_str import write2mdb,insert_mdb
	if funcN in globals():
		funcArg=globals()[funcN]
	else:
		return {}
	df = funcArg(tkLst,debugTF=debugTF,dbname=dbname,**optx)
	try:
		#dbM = conn2mgdb(dbname=dbname)
		#ordered=False # to insert whatever is available
		#if not tablename in dbM.collection_names():
		#	zsrt=[1]*len(zpk)
		#	dbM[tablename].create_index([(k,v) for k,v in zip(zpk,zsrt)],unique=True)
		#ret=dbM[tablename].insert_many(df,ordered=ordered)
		dbM=None;zpk=zpk
		mobj,dbM,err_msg =upsert_mdb(df,dbM,dbname=dbname,tablename=tablename,zpk=zpk,insertOnly=True)
	except Exception as e:
		sys.stderr.write("**ERROR: {}:{}\n".format("func2mdb()",str(e)))
	return df	
def renewChk(pbdtCurr,pbdtMod,deltaTolerance=86400):
	deltaPassed=pd.Timedelta(pbdtCurr - pbdtMod).total_seconds()
	sys.stderr.write(" --curr:{},last:{}:deltaPassed:{}\n".format(pbdtCurr,pbdtMod,deltaPassed))
	return deltaPassed>deltaTolerance

def lastRunChk(objChk={},tableChk='',deltaTolerance=43200,clientM=None,dbname='ara',**optx):
	pbdtCurr=pd.datetime.now()
	lastObj,clientM,_=find_mdb(objChk,clientM=clientM,dbname=dbname,tablename=tableChk,limit=1)
	if not lastObj:
		pbdtMod=pbdtCurr
		renewTF=True
	else:
		pbdtMod=lastObj[0]['pbdt']
		renewTF=renewChk(pbdtCurr,pbdtMod,deltaTolerance)
		if renewTF:
			pbdtMod=pbdtCurr
	objChk.update(pbdt=pbdtMod)
	return renewTF,objChk,clientM

def batchOTF(funcArg,tkLst=[],tableChk='',zpkChk=["ticker"],deltaTolerance=43200,**optx):
	for ticker in tkLst: 
		sys.stderr.write("==Batch Running:{} on {}\n".format(ticker,funcArg))
		runOTF(funcArg,ticker,tableChk,zpkChk=zpkChk,deltaTolerance=deltaTolerance,**optx)

@safeRunArg([])
def runOTF(funcArg,ticker='',tableChk='',zpkChk=["ticker"],deltaTolerance=43200,**optx):
	'''
	To  ticker='AAPL'
	where real-time data is only grab based on 'deltaTolerance' in seconds
	current setup is half-days
	'''

	if isinstance(ticker,list):
		tkLst=ticker
		dLst=[]
		for tkX in tkLst: 
			sys.stderr.write("==BATCH Running:{} on {}\n".format(ticker,funcArg))
			dd = runOTF(funcArg,tkX,tableChk,zpkChk=zpkChk,deltaTolerance=deltaTolerance,**optx)
			if isinstance(dd,list):
				dLst.extend(dd)
			else:
				dLst.extend([dd])
		return dLst

	if isinstance(funcArg,str):
		if funcArg in globals() and hasattr(globals()[funcArg],'__call__'):
			funcArg =  globals()[funcArg]
		else:
			return []
	sys.stderr.write("==START Running:{} on {}\n".format(ticker,funcArg))
	dbname=getKeyVal(optx,'dbname','ara')
	optx.update({'dbname':dbname})
	tablename=getKeyVal(optx,'tablename','')
	if not all([tablename, ticker]):
		return []
		
	if 'modules' in optx:
		objChk=dict(ticker=ticker,module=optx['modules'])
	else:
		objChk=dict(ticker=ticker)
	if not tableChk:
		tableChk=tablename+'_chk'
	renewTF,objChk,clientM = lastRunChk(objChk=objChk,tableChk=tableChk,deltaTolerance=deltaTolerance,**optx)
	if renewTF:
		sys.stderr.write("==Data outdated or never run, Running:{}\n".format(funcArg))
		retObj = funcArg(ticker,**optx)
		if len(retObj)<1:
			return []

		xChk={ky:val for ky,val in retObj[0].items() if ky in zpkChk}
		objChk.update(xChk)
		retObj,clientM,errChk = upsert_mdb(retObj,**optx)
		sys.stderr.write(" --Update {} to {}\n".format(objChk,tableChk))
		objChk,clientM,errChk = upsert_mdb(objChk,clientM=clientM,tablename=tableChk,zpk=zpkChk)
	else:
		sys.stderr.write("==Data exist, LoadFromTable:{}\n".format(tablename))
		objChk.pop('pbdt',None)
		optx.pop('zpk',None)
		retObj,clientM,errMsg = find_mdb(objChk,clientM=clientM,**optx)
	return retObj

def main_tst():
	args = sys.argv[1:]
	if len(args)<1:
		#exit(0)
		args=['AAPL','IBM']
	elif len(args)==1 and ',' in args[0]:
		args = args[0].split(',')
	elif len(args)==1 and '-' in args[0]:
		args = sys.stdin.read().strip().split()
	#df = yh_spark_hist(args)
	#df = yh_hist_query(args,types='spark',range='1d',pchgTF=False)
	#df = yh_hist_query(args,types='quote',pchgTF=False,filter='symbol,regularMarketPrice,regularMarketChange,regularMarketOpen,regularMarketDayHigh,regularMarketDayLow,regularMarketVolume')
	#df = yh_hist_query(args,types='quote')
	df = yh_quote_curr(args)
	if isinstance(df,list):
		sys.stderr.write("\n{}".format(df))
	else:
		sys.stderr.write("\n{}".format(df.to_string()))

if __name__ == '__main__':
	main_tst()
