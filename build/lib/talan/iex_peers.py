#!/usr/bin/env python3
# get peer group based on ticker
# Usage of:
# python -c "from iex_peers import ticker2sectorRanking as t2sr;print t2sr(ticker='AMZN',sector='Consumer Discretionary')"
# python -c "from iex_peers import peers_performance as ppf;print ppf(tkLst=['AMZN','BABA','JD','EBAY','MELI','ORLY','XLY'])"

import sys
import numpy as np
import pandas as pd
from _alan_calc import sqlQuery

def df2out(df,output='',indexTF=False,sep='|',orient='records',formatters=None,float_format=None,justify=None,border=None,table_id=None,columns=None,force_ascii=False):
	""" datafame to alternative output format
	"""
	if output=='csv':
		return df.to_csv(index=indexTF,sep=sep)
	elif output=='json': # orient: [split|records|index|columns|values]
		return df.to_json(orient=orient,force_ascii=force_ascii).encode('utf8')
	elif output=='dict': # orient: [dict|list|series|split|records|index]
		return df.to_dict(orient=orient)
	elif output=='html':
		return df.to_html(index=indexTF,formatters=formatters,
			float_format=float_format,justify=justify,border=border,
			table_id=table_id,columns=columns)
	else:
		return df

def peers_performance(tkLst=[],xqTmp='',debugTF=True):
	'''
	return peRatio's for comparison based on tickers: 'tkLst'
	'''
	if not tkLst:
		return []
	if not xqTmp:
		xqTmp="""select ticker,"changePercent"/100.0 as pchg, "trailingPE" as "peRatio" from yh_quote_curr where ticker in ('{}') ORDER BY pchg DESC"""
	xqr = xqTmp.format("','".join(tkLst) )
	ret = sqlQuery(xqr)
	#ret = ret.dropna()
	return ret

def merge_ticker_cik(tkLst=[],colStr='ticker,company_cn',dbname='ara',tablename='mapping_ticker_cik'):
	xqTmp="""Select {colStr} from "{tablename}" where ticker in ('{tkQStr}')"""
	tkQStr="','".join(tkLst)
	xqr = xqTmp.format(**locals())
	df = sqlQuery(xqr,dbname=dbname)
	return df

def ticker2sectorRanking(ticker,sector='Information Technology',xqTmp='',debugTF=True):
	'''
	return ranking of 'ticker' in the 'sector' 
	'''
	if not xqTmp:
		xqTmp="""Select ticker,"marketCap" from yh_quote_curr where  "marketCap">=(SELECT "marketCap" FROM yh_quote_curr WHERE ticker='{ticker}') and ticker in (SELECT ticker from mapping_ticker_cik where sector='{sector}') ORDER BY "marketCap" DESC"""
	xqr = xqTmp.format(ticker=ticker,sector=sector)
	ret = sqlQuery(xqr)
	return ret

def ticker_peers(ticker,xqTmp='',debugTF=True,j=0):
	'''
	USE
	SELECT a.*,b."marketCap" as marketCap, b."trailingPE" as peRatio FROM (select ticker,sector,industry from "yh_summaryProfile" where industry in (select industry from "yh_summaryProfile" where ticker='{}')) as a, yh_quote_curr b WHERE a.ticker=b.ticker  ORDER BY marketCap DESC
	'''
	if len(xqTmp)<1:
		xqLst=(
		"""SELECT a.*,b."marketCap" as marketCap, b."trailingPE" as peRatio FROM (select ticker,sector,industry from "yh_summaryProfile" where industry in (select industry from "yh_summaryProfile" where ticker='{}')) as a, yh_quote_curr b WHERE a.ticker=b.ticker  ORDER BY marketCap DESC""",
		"""SELECT a.*,b."marketCap" as marketCap, b."trailingPE" as peRatio FROM (select ticker,sector,industry from "yh_summaryProfile" where sector in (select sector from "yh_summaryProfile" where ticker='{}')) as a, yh_quote_curr b WHERE a.ticker=b.ticker  ORDER BY marketCap DESC""",
		"""SELECT a.*,b.\"marketCap\" as marketCap, b.\"trailingPE\" as peRatio FROM (SELECT etfname as ticker,sector_alias as sector,NULL as industry FROM spdr_sector WHERE sector_alias in (select sector from "yh_summaryProfile" where ticker='{}' )) as a, yh_quote_curr b WHERE a.ticker=b.ticker  ORDER BY marketCap DESC""")
	xqTmp = xqLst[j]
	xqr=xqTmp.format(ticker)
	if debugTF is True:
		sys.stderr.write(xqr+"\n")
	ret = sqlQuery(xqr)
	if len(ret)<5 and j<len(xqLst)-1:
		ret=ticker_peers(ticker,debugTF=debugTF,j=j+1)
	return ret

def get_peers(ticker,xqTmp='',top=6,lb=-2,nb=5,debugTF=True):
	try:
		ret = ticker_peers(ticker,debugTF=debugTF)
	except Exception as e:
		sys.stderr.write("**ERROR:{}\n".format(str(e)))
		return {}
	if len(ret)<1:
		return {}
	idx = ret.query('ticker=={!r}'.format(ticker)).index[0]
	a = np.arange(top)
	b = max(0,idx+lb)+np.arange(nb)
	c = np.unique(np.concatenate((a, b)))
	d = np.intersect1d(ret.index,c)
	dx1 = ret.loc[d,:]
	# add sector etf peer
	dx2 = ticker_peers(ticker,debugTF=debugTF,j=2) # apply sector directly
	df = pd.concat([dx1,dx2],ignore_index=False)
	df['idx']=0
	df.reset_index(drop=True,inplace=True)
	idx = df.query('ticker=={!r}'.format(ticker)).index[0]
	df.loc[idx,'idx']=1
	df.rename(columns={'ticker':'peers'},inplace=True)
	df['ticker'] = ticker
	if debugTF is True:
		sys.stderr.write("{}".format((a,b,c,d)))
		sys.stderr.write("{}".format(df))
	return df

def yh_peers(tkLst=[],debugTF=False,peersLstTF=False):
	""" Get peer group based on tickers: 'tkLst'
	"""
	if not isinstance(tkLst,list):
		tkLst=[tkLst]
	if len(tkLst)<1:
		sys.stderr.write("tkLst:{} is empty.\n".format(tkLst))
		return {} if peersLstTF else ({},{},{})
	df=pd.DataFrame()
	peerLst=[]
	for tkX in tkLst:
		dx = get_peers(tkX,debugTF=debugTF)
		if len(dx)<1:
			continue
		xp = { 'peers' :[str(x) for x in dx['peers'].values if x!=tkX] }
		xp.update(ticker=tkX)
		peerLst.append(xp)
		df = pd.concat([df,dx],ignore_index=False)
	if 'peers' not in df:
		ret= [] if peersLstTF else df,[],{}
		return ret
	tqLst=df['peers'].values
	dq = merge_ticker_cik(tqLst)
	if len(dq)>0:
		dq.rename(columns={'ticker':'peers'},inplace=True)
		df = df.merge(dq,on='peers',how='left')
	if peersLstTF:
		return peerLst
	if len(df)>0: 
		peerInfo=dict(peersList=','.join(df['peers'].values),peers=list(df['peers']))
		peerInfo.update(peers_cn=list(df['company_cn']))
	else :
		peerInfo=dict(peersList='',peers=[],peers_cn=[])
	return df,peerLst,peerInfo

def iex_peers(tkLst=[],debugTF=False,peersLstTF=False):
	""" Get peer group based on tickers: 'tkLst'
	passthru of yh_peers()
	help(yh_peers)
	"""
	return yh_peers(tkLst=tkLst,debugTF=debugTF,peersLstTF=peersLstTF)

if __name__ == '__main__':
	from _alan_optparse import parse_opt, subDict
	opts, args = parse_opt(sys.argv)
	df,peerLst,peerInfo = yh_peers(args,debugTF=opts['debugTF'])
	if opts['output'] in ['csv','html','dict']:
		sys.stdout.write("{}".format(df2out(df,output=opts['output'])))
	else:
		sys.stdout.write("{}".format(peerLst))
