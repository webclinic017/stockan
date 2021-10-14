#!/usr/bin/env python
''' Description: grab info/history from available stock symbols and save them to iex_{types}_temp
	then update into iex_{types}_hist table
	cronjob every Saturday
	available types are [financials,earnings,chart,dividends,company,quote,stats]
    Usage of:
	python iex_types_batch.py --types=dividends,financials --range=5y --no_active --hist_upd
	python iex_types_batch.py --types=stats --no_active --hist_upd
	python iex_types_batch.py --types=chart --range=5d --no_active
	python iex_types_batch.py --types=quote --no_active # Using the iex company list
	python iex_types_batch.py --types=quote IBM AAPL --hist_upd # Update to hist table
	python iex_types_batch.py --date=20181001 
	python iex_types_batch.py --types=chart --range=1d # Same as --date=DATE (today) for minute types 
	python iex_types_batch.py --types=chart --range=1d --extra_param=chartLast=1 IBM# Same as --date=DATE (today) for minute types 
    Cronjob:
	python iex_types_batch.py --types=quote --no_active --hist_upd # for daily update Mon-Fri @ 4:30pm
	python iex_types_batch.py --types=chart --no_active --hist_upd # for weekly update Sat @ 5am
    Ref: https://iextrading.com/developer/docs/#batch-requests
    Note: dividends data are not up-to-date, e.g., IBM has more than 6mo lag.
    Last mod., Tue Jan 22 16:08:12 EST 2019
	add earnings and peers info into mongoDB
'''
import sys
from optparse import OptionParser
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pymongo import MongoClient
import datetime
import requests
from pandas.io.json import json_normalize
import json
from _alan_str import write2mdb
from _alan_calc import upd_temp2hist

def printq(*args,**kwargs):
	sep=kwargs.pop('sep',' ')
	end=kwargs.pop('end','\n')
	file=kwargs.pop('file',sys.stdout)
	flush=kwargs.pop('flush',False)
	n=len(args)-1
	for j,x in enumerate(args):
		y = end if j==n else sep
		file.write("{}{}".format(x,y))
	if flush:
		file.flush()
	return n+1

def epoch_parser(x,s=1000): return datetime.datetime.fromtimestamp(int(x/s))
def ymd_parser(x,fmt='%Y%m%d'): return datetime.datetime.strptime(str(x),fmt)

#DEPRECATED, use _alan_calc.upd_temp2hist instead
def sql_temp2hist(pgDB=None,temp="iex_earnings_temp",hist="iex_earnings_hist",col1="ticker",col2="pbdate"):
	"""
	Insert/update additional values from table: [temp] to [hist]
	base on primary keys: [col1,col2]
	DEPRECATED, use _alan_calc.upd_temp2hist instead
	"""
	if pgDB is None:
		return None
	xqTmp='''CREATE TABLE IF NOT EXISTS {hist} AS SELECT * FROM {temp} WHERE 1=2;
	DELETE FROM {hist} B USING {temp} C WHERE B."{col1}" = C."{col1}" AND B."{col2}" = C."{col2}";
	INSERT INTO {hist} SELECT DISTINCT * FROM {temp}
	'''
	dux=locals()
	xqr = xqTmp.format(**dux)
	pgDB.execute(xqr,pgDB)
	return pgDB

def assign_type_spec(typx):
	earningsTyp={"type":"earnings","dateColName":"EPSReportDate","colLst":["src","ticker","pbdate","estimatedEPS","yearAgo","consensusEPS","fiscalPeriod","EPSReportDate","announceTime","EPSSurpriseDollar","estimatedChangePercent","symbolId","numberOfEstimates","actualEPS","fiscalEndDate","yearAgoChangePercent"],"colTyp":[str, str, int, float, float, float, str, str, str, float, float, int, float, float, str, float]}
	financialsTyp={"type":"financials","dateColName":"reportDate","colLst":["src","ticker","pbdate","cashChange","cashFlow","costOfRevenue","currentAssets","currentCash","currentDebt","grossProfit","netIncome","operatingExpense","operatingGainsLosses","operatingIncome","operatingRevenue","reportDate","researchAndDevelopment","shareholderEquity","totalAssets","totalCash","totalDebt","totalLiabilities","totalRevenue","freq"],"colTyp": [str, str, int, float, float, float, float, float, float, float, float, float, float, float, float, str, float, float, float, float, float, float, float, str]}
	chartTyp={"type":"chart","dateColName":"date","colLst":["src","ticker","pbdate","name","open","high","low","close","change","changeOverTime","changePercent","date","label","unadjustedVolume","volume","vwap","adjusted"]}
	minuteTyp={"type":"chart","dateColName":"minute","colLst":["src","ticker","pbdate","name","high","marketVolume","notional","marketLow","close","changeOverTime","open","marketNumberOfTrades","label","low","marketHigh","marketOpen","volume","date","marketNotional","minute","average","marketClose","marketAverage","marketChangeOverTime","numberOfTrades","epochs"]}
	dividendsTyp={"type":"dividends","dateColName":"exDate","colLst":["src","ticker","pbdate","declaredDate","qualified","exDate","indicated","type","recordDate","flag","amount","paymentDate"]}
	quoteTyp={"type":"quote","dateColName":"latestUpdate","colLst":["src","ticker","pbdate","name","sector","avgTotalVolume","iexAskSize","companyName","week52High","symbol","marketCap","previousClose","high","iexVolume","iexMarketPercent","changePercent","latestSource","iexLastUpdated","close","calculationPrice","latestVolume","open","delayedPriceTime","change","extendedPrice","week52Low","latestUpdate","openTime","latestPrice","latestTime","iexAskPrice","peRatio","iexRealtimeSize","closeTime","extendedPriceTime","low","iexBidSize","delayedPrice","iexRealtimePrice","iexBidPrice","extendedChangePercent","ytdChange","primaryExchange","extendedChange","volume","adjusted"],"colTyp":[str,str,int,str,str,int,float,str,float,str,int,float,float,float,float,float,str,float,float,str,int,float,int,float,float,float,int,int,float,str,float,float,float,int,float,float,float,int,float,float,float,float,str,float,int,float]}
	companyTyp={"type":"company","dateColName":"","colLst":["src","ticker","CEO","companyName","description","exchange","industry","issueType","sector","symbol","tags","website"]}
	peersTyp={"type":"peers","dateColName":"","colLst":["src","ticker","peers","pbdate"]}
	statsTyp={"type":"stats","dateColName":"latestEPSDate","colLst":["src","ticker","shortRatio", "returnOnEquity", "revenue",  "revenuePerEmployee", "month1ChangePercent", "consensusEPS", "latestEPS", "shortInterest", "week52high", "year2ChangePercent", "marketcap", "peRatioLow", "shortDate", "latestEPSDate", "EBITDA", "day5ChangePercent", "day50MovingAvg", "EPSSurpriseDollar", "revenuePerShare", "profitMargin", "peRatioHigh", "ttmEPS", "week52low", "grossProfit", "institutionPercent", "returnOnAssets", "priceToSales", "week52change", "dividendRate", "priceToBook", "companyName", "symbol", "ytdChangePercent", "dividendYield", "insiderPercent", "month6ChangePercent", "beta", "month3ChangePercent", "sharesOutstanding", "day200MovingAvg", "returnOnCapital", "year1ChangePercent", "debt", "EPSSurprisePercent", "cash", "numberOfEstimates", "exDividendDate", "day30ChangePercent", "year5ChangePercent"]}
	typeSpec={"earnings":earningsTyp,"financials":financialsTyp,"chart":chartTyp,"dividends":dividendsTyp,"quote":quoteTyp,"company":companyTyp,"minute":minuteTyp,"stats":statsTyp,"peers":peersTyp}
	dateColName=typeSpec[typx]["dateColName"]
	colLst=typeSpec[typx]["colLst"]
	colTyp=typeSpec[typx]["colTyp"] if 'colTyp' in typeSpec[typx] else None
	return(typx,dateColName,colLst,colTyp)
	
def reshape_iex_typx(ticker,typx,jdTmp,tsTF=False,debugTF=False,period='quarterly'):
	# ['earnings','financials'] contain different json structure
	if typx in ['earnings','financials']:
		jdX=jdTmp[ticker][typx]
		if len(jdX)<1:
			return(typx,pd.DataFrame())
		else:	
			jdX=jdX[typx]
	elif typx in ['peers']:
		jdX=jdTmp[ticker]
	elif typx in ['minute']:
		jdX=jdTmp
	else:
		jdX=jdTmp[ticker][typx]
	if len(jdX)<1:
		return(typx,pd.DataFrame())
	dx=json_normalize(jdX)
	if 'ticker' not in dx:
		dx['ticker']=ticker
	dx['src']='iex'
	# use minute type for minute-by-minute quote is [minute] column is in the data 
	(typy,dateColName,colLst,colTyp) = assign_type_spec('minute' if 'minute' in dx else typx)
	if debugTF is True:
		printq("typy:{}\ndateColName:{}\ncolLst:{}\n".format(typy,dateColName,colLst),file=sys.stderr)
		printq("Before adusted: ",dx.iloc[-1].to_dict(),file=sys.stderr)
	if typx in ['chart','quote','minute']:
		dx['name']=ticker.replace(".","-") if ticker!='AGM.A' else ticker
		if typx == 'quote' and dx['closeTime'].iloc[-1]<dx['openTime'].iloc[-1]:
			dx['close'] = dx['latestPrice'].copy()
		if 'close' not in dx:
			dx['close']=None
		if 'adjusted' not in dx:
			dx['adjusted']=dx['close'] 
		dx['volume']=dx['avgTotalVolume'].astype(int) if typx=='quote' else dx['volume'].astype(int)
	elif typx in ['peers']: # add 'pbdate' to up-to-date column
		dx["pbdate"]=int(datetime.datetime.now().strftime("%Y%m%d"))
	elif typx in ['financials']: # add additional column: period 
		dx["freq"]=period[:1].upper()

	try:
		if len(dateColName)>0:
			if 'time' in dateColName[-4:].lower() or 'latestUpdate' in dateColName:
				pbdatetime = dx[dateColName].apply(epoch_parser)
				dx["pbdate"]=pbdatetime.apply(lambda x:x.strftime("%Y%m%d")).astype(int)
			elif dateColName == 'minute':
				dx["pbdate"]=[x.replace('-','') for x in dx['date'] ]
				dx["epochs"]=[int(datetime.datetime.strptime(y+x,'%Y%m%d%H:%M').strftime("%s")+'000')for x,y in zip(dx[dateColName],dx['pbdate']) ]
				dx["pbdate"]=dx['pbdate'].astype(int)
				pbdatetime = dx["epochs"].apply(epoch_parser)
			else:
				pbdatetime=dx[dateColName].apply(lambda x:ymd_parser(x,'%Y-%m-%d'))
				dx["pbdate"]=pbdatetime.apply(lambda x:x.strftime("%Y%m%d")).astype(int)
	except Exception as e:
		printq("**ERROR: {}, assign pbdate=20010101".format(str(e)),file=sys.stderr)
		dx["pbdate"]=20010101
	for j,xc in enumerate(dx.columns):
		if str(dx[xc].iloc[0]).lower()=='nan':
			dx[xc].iloc[0] = np.nan 

	if tsTF is True and typx in ['chart','quote','minute'] and len(dateColName)>0:
		dx.set_index(pd.DatetimeIndex(pbdatetime),inplace=True)
		dx.index.rename('date',inplace=True)
	for ky in set(colLst)-set(dx.columns):
		dx[ky] = None
	if debugTF is True:
		printq("After adusted: ",dx.iloc[-1].to_dict(),file=sys.stderr)
	if 'close' in colLst:
		dx.dropna(subset=['close'],inplace=True)
	if debugTF:
		printq(type(dx),file=sys.stderr)
		printq(colLst,file=sys.stderr)
	da = dx[colLst]	
	if debugTF:
		printq(type(da),file=sys.stderr)
		printq(da.tail().to_csv(sep='|',index=False),file=sys.stderr)

	if colTyp is not None:
		#printq("Assign {} type to {}".format(typx,colTyp),file=sys.stderr)
		try:
			da = da.astype( dict(zip(colLst,colTyp)) )
		except Exception as e:
			printq(da.head().to_csv(sep='|',index=False),file=sys.stderr)
			for j,(c,t) in enumerate(zip(colLst,colTyp)):
				try:
					if t in [int,np.integer,long] and da[c] is None:
						da[c]=0
					da[c].astype(t)
				except Exception as e:
					printq( "**ERROR: {} at assigning {}.{}:{} to type {}".format(str(e),j,c,da[c].values,t),file=sys.stderr)
					if t == float:
						da[c]=None
					elif t in [int,np.integer,long]:
						da[c]=0
					da[c].astype(t)
	return(typy,da)

def chart2price(typy,dx,saveDB,pgDB,rmode,debugTF=False):
	if typy not in ['chart','quote']:
		return None
	table1="prc_temp_iex"
	#if typy == 'quote' and dx['closeTime'].iloc[-1]<dx['openTime'].iloc[-1]:
	#	da = dx[["open","high","low","latestPrice","volume","adjusted","pbdate","name"]].copy()
	#	da.rename(columns={"latestPrice":"close"},inplace=True)
	#	da["adjusted"] = da["close"].copy()
	#else:
	#	da = dx[["open","high","low","close","volume","adjusted","pbdate","name"]]
	da = dx[["open","high","low","close","volume","adjusted","pbdate","name"]]
	if debugTF is True and da.shape[0]>0 :
		printq( "debug @ chart2price():\n",da.tail(2).to_csv(sep='|',index=False),file=sys.stderr)

	if saveDB is True:
		da.to_sql(table1,pgDB,schema='public',index=False,if_exists=rmode)
	else:
		printq( da.head(2).to_csv(sep='|',index=False),file=sys.stderr)
		printq( da.tail(2).to_csv(sep='|',index=False),file=sys.stderr)
	return typy

# also see _alan_str.df_tofile
def df_tofile(fp,df,output=None,sep='|',indexTF=False,hdrTF=True):
	"""
	Write dataframe:df to file via file handle:fp
	where
		fp: file handle
		df: data in dataframe
		output: output format of [csv|json|html|string|None], optional
		sep: output separator, default to '|', optional
		indexTF: flag to show index, default to False, optional
		hdrTF: flag to show header, default to True, optional
	"""
	ret = df_output(df,output=output,sep=sep,indexTF=indexTF,hdrTF=hdrTF)
	str_tofile(fp,ret)

def str_tofile(fp,s):
	"""
	Write string:s to file via file handle:fp
	"""
	if isinstance(fp,str) is True:
		fp = open(fp,'w') if fp != '-' else sys.stdout
	fp.write(s)

def df_output(df,output=None,sep='|',indexTF=False,hdrTF=True):
	"""
	convert dataframe:df to a string 
	where
		df: data in dataframe
		output: output format of [csv|json|html|string|None], optional
		sep: output separator, default to '|', optional
		indexTF: flag to show index, default to False, optional
		hdrTF: flag to show header, default to True, optional
	"""
	ret=''
	if output == 'csv':
		ret = df.to_csv(sep=sep,index=indexTF,header=hdrTF)
	elif output == 'json':
		ret = df.to_json(orient='records')
	elif output == 'html':
		ret = df.to_html(index=indexTF,header=hdrTF)
	elif output == 'string':
		ret = df.to_string(index=indexTF,header=hdrTF)
	return ret

def reshape_iex_types(ticker,types,jdTmp,saveDB,pgDB,rmode,debugTF=False,hdrTF=True,output=None,sep='|',indexTF=False,tsTF=False,clientM=None,period='quarterly'):
	dbscm='public';dbidx=False
	for typx in list(map(str.strip,types.split(','))) :
		try:
			(typy,da)=reshape_iex_typx(ticker,typx,jdTmp,tsTF=tsTF,debugTF=debugTF,period=period)
		except Exception as e:
			printq( "**ERROR: {} of {}/{} @ reshape_iex_typx()".format(str(e),ticker,typx),file=sys.stderr)
			continue
		if len(da)<1:
			printq( "**No data for {}/{}".format(ticker,typx),file=sys.stderr)
			continue
		chart2price(typy,da,saveDB,pgDB,rmode,debugTF=debugTF)
		table1="iex_{}_temp".format(typy)
		if debugTF is True:
			printq( "debug @ reshape_iex_types():\n",da.tail(2).to_csv(sep='|',index=False),file=sys.stderr)
		if saveDB is True:
			da.to_sql(table1,pgDB,schema=dbscm,index=dbidx,if_exists=rmode)
			if typx in ['quote','peers','earnings','financials','stats']:
				tbhist="iex_{}_hist".format(typy)
				if typx=='financials':
					zpk = {'ticker','pbdate','freq'} 
				elif typx=='quote':
					#zpk = {'ticker','latestUpdate'} 
					zpk = {'ticker'}
				else:
					zpk = {'ticker','pbdate'} 
				write2mdb(da,clientM=clientM,tablename=tbhist,zpk=zpk)
		else:
			printq( da.shape,file=sys.stderr)
		df_tofile(sys.stdout,da,output=output,sep=sep,indexTF=indexTF,hdrTF=hdrTF)
	return (da, typy)

def get_tkLst_iex(pgDB,activeON):
	df=pd.read_json("https://api.iextrading.com/1.0/ref-data/symbols")
	if activeON is True: # apply only to act_code=1 tickers
		m=pd.read_sql('SELECT distinct ticker FROM mapping_ticker_cik WHERE act_code=1',pgDB)
		mlist=[ x.replace("-",".") for x in m.ticker ]
		tkLst=list(df[['symbol']].query('symbol=={}'.format(mlist)).symbol)
	else:
		tkLst=list(df['symbol'])
	return tkLst

def iex_types_batch(args=[],temphistTF=False,wmode='replace',extraParam=None,ranged='5d',debugTF=False,tsTF=True,hostname='localhost',activeON=True,period='quarterly',saveDB=True,indexTF=False,date=None,output=None,sep='|',dbname='ara',types='chart',**optx):
	''' grab past 1y history from available stock symbols and save them to iex_{types}_temp
		then update into iex_{types}_hist table
	'''
	if debugTF is True:
		printq( sorted(locals().keys()),file=sys.stderr)
		printq( [dbname,hostname,period,saveDB,types,wmode],file=sys.stderr)
	pgDB = create_engine('postgresql://sfdbo@{}:5432/{}'.format(hostname,dbname))
	mgDB =MongoClient("{}:27017".format(hostname))[dbname]
	tkLst=get_tkLst_iex(pgDB,activeON) if len(args)<1 else sys.stdin.read().strip().split("\n") if args[0]=='-' else args
	tkLst=['BRK.B' if tx=='BRKB' else 'BF.B' if tx=='BFB' else tx.replace('-','.') for tx in tkLst]
	chartIEXOnly=True
	token='sk_c3846ce073c849f5838e5ae0be3d005d'
	if date is not None:
		utmp='https://cloud.iexapis.com/stable/stock/{{}}/chart/date/{date}?token={token}&chartIEXOnly={chartIEXOnly}&'
		urx=utmp.format(token=token,chartIEXOnly=chartIEXOnly,date=date)
		#urx="https://api.iextrading.com/1.0/stock/{{}}/chart/date/{}".format(date)
		types='minute'
		n=1
	else:
		utmp='https://cloud.iexapis.com/stable/stock/market/batch?token={token}&chartIEXOnly={chartIEXOnly}&'
		utmp=utmp.format(token=token,chartIEXOnly=chartIEXOnly)
		urx=utmp+"&symbols={}&types={}&range={}&period={}"
		#urx="https://api.iextrading.com/1.0/stock/market/batch?symbols={}&types={}&range={}&period={}"
		n=100
	tkM = [tkLst[i * n:(i + 1) * n] for i in range((len(tkLst) + n - 1) // n )]
	da=pd.DataFrame()
	rmode=wmode
	for j,tkTmp  in enumerate(tkM):
		tkStr=','.join(tkTmp)
		url=urx.format(tkStr,types,ranged,period)
		if extraParam is not None:
			url="{}{}{}".format(url,'?' if types=='minute' else '&', extraParam)
		if debugTF is True:
			printq("===RUNNING {}:{}\n\tURL:{}".format(j,tkStr,url),file=sys.stderr)
		try:
			ret=requests.get(url)
			jdTmp=ret.json()
		except Exception as e:
			printq("**ERROR {}:{}\n\tURL:{}".format(j,tkStr,url),file=sys.stderr)
			continue
		if debugTF is True:
			printq( jdTmp,file=sys.stderr)
		for jk, ticker  in enumerate(tkM[j]):
			printq("===RUNNING {}:{}".format(j*n+jk+1,ticker),file=sys.stderr)
			try:
				hdrTF = True if (jk+j)<1 else False
				da, typy = reshape_iex_types(ticker,types,jdTmp,saveDB,pgDB,rmode,debugTF=debugTF,hdrTF=hdrTF,output=output,sep=sep,indexTF=indexTF,tsTF=tsTF,clientM=mgDB,period=period)
				rmode = "append"
			except Exception as e:
				printq( str(e), '**ERROR {}. {}'.format(j*n+jk,ticker),file=sys.stderr)
				continue
	if saveDB is False:
		return da, None
	if temphistTF is False:
		return da, pgDB
	for typx in list(map(str.strip,types.split(','))) :
		typy='minute' if typx=='chart' and ranged=='1d' else typx
		table1="iex_{}_temp".format(typy)
		temp=table1;hist=temp.replace('temp','hist')
		col2='symbol' 
		if typy=='company':
			col2 = "symbol"
		elif typy=='minute':
			col2 = "epochs"
		elif typy=='quote':
			col2 = "latestUpdate"
		else:
			col2 = "pbdate"
		pcol = ['ticker',col2]
		if typy=='financials':
			pcol = pcol+['freq']
		#sql_temp2hist(pgDB,temp=temp,hist=hist,col2=col2)
		xqr = upd_temp2hist(pgDB,temp=temp,hist=hist,pcol=pcol)
		if debugTF:
			printq("Update table {} to {}".format(temp,hist),file=sys.stderr)
			printq("sql:\n{}".format(xqr),file=sys.stderr)
		if typx in ['chart','quote']:
			table2="prc_temp_iex"
			temp=table2;hist=temp.replace('temp','hist')
			#sql_temp2hist(pgDB,temp=temp,hist=hist,col1='name')
			upd_temp2hist(pgDB,temp=temp,hist=hist,pcol=['name','pbdate'])
	return da, pgDB

def opt_iex_types(argv,retParser=False):
	""" command-line options initial setup
	    Arguments:
		argv:   list arguments, usually passed from sys.argv
		retParser:      OptionParser class return flag, default to False
	    Return: (options, args) tuple if retParser is False else OptionParser class
	"""
	parser = OptionParser(usage="usage: %prog [option] [SYMBOL1 SYMBOL2...]", version="%prog 0.62",
		description="Pull up-to-date stock financials/quote from IEX")
	parser.add_option("-r","--range",action="store",dest="ranged",default="5d",
		help="range [5d,1m,3m,6m,ytd,1y,2y,5y] (default: 5d) Note: ONLY for [chart,dividends] types")
	parser.add_option("","--date",action="store",dest="date",
		help="date for intraday minute data (default: None) Note: Same as [chart] types with [1d] range. Data are saved to iex_minute_temp table")
	parser.add_option("","--period",action="store",dest="period",default="quarterly",
		help="period [annual,quarterly] (default: quarterly) Note: ONLY for [financials] types")
	parser.add_option("","--types",action="store",dest="types",default="chart",
		help="Comma delimited list of types [chart,company,dividends,earnings,financials,quote,stats] (default: chart)")
	parser.add_option("-d","--database",action="store",dest="dbname",default="ara",
		help="database (default: ara)")
	parser.add_option("","--host",action="store",dest="hostname",default="localhost",
		help="db host (default: localhost)")
	parser.add_option("-w","--wmode",action="store",dest="wmode",default="replace",
		help="db table write-mode [replace|append] (default: replace)")
	parser.add_option("","--no_database_save",action="store_false",dest="saveDB",default=True,
		help="no save to database (default: save to database)")
	parser.add_option("","--hist_upd",action="store_true",dest="temphistTF",default=False,
		help="update history from iex_[type]_temp to iex_[type]_hist (default: False)")
	parser.add_option("-o","--output",action="store",dest="output",
		help="OUTPUT type [csv|html|json] (default: no output)")
	parser.add_option("","--no_datetimeindex",action="store_false",dest="tsTF",default=True,
		help="no datetime index (default: use datetime)")
	parser.add_option("","--show_index",action="store_true",dest="indexTF",default=False,
		help="show index (default: False) Note, OUTPUT ONLY")
	parser.add_option("-s","--sep",action="store",dest="sep",default="|",
		help="output field separator (default: |) Note, OUTPUT ONLY")
	parser.add_option("","--no_active",action="store_false",dest="activeON",default=True,
		help="apply to all IEX symbols(default: active symbols ONLY)")
	parser.add_option("","--extra_param",action="store",dest="extraParam",
		help="additional parameters to IEX (default: None). Note: May cause unexpected error!")
	parser.add_option("","--debug",action="store_true",dest="debugTF",default=False,
		help="debugging (default: False)")
	(options, args) = parser.parse_args(argv[1:])
	if retParser is True:
		return parser
	return (vars(options), args)

if __name__ == '__main__':
	opts,args =opt_iex_types(sys.argv)
	da, pgDB=iex_types_batch(args,**opts)
	if pgDB is not None:
		pgDB.dispose()
