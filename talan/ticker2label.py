#!/usr/bin/env python
''' get basic info from a list of tickers 'tkLst'
required tables:
  mapping_ticker_cik (pgDB)
  spdr_sector
  sector_alias_industry
  yh_curr_quote
  yh_summaryProfile
  qS_keyStatistics (TBD)

Usage of 
python3 -c "from ticker2label import t2l;d=t2l('QQQ');print(d)"

'''
# -*- coding: utf-8 -*-
import sys,os

from _alan_calc import sqlQuery,conn2pgdb,getKeyVal,renameDict
import pandas as pd
if sys.version_info.major == 2:
	reload(sys)
	sys.setdefaultencoding('utf8')

def tk2infoM(ticker='',tablename='yh_curr_quote',colx='ticker',dbname='ara',clientM=None,**optx):
	from yh_chart import runOTF;
	from _alan_str import find_mdb
	funcArg,zpk,deltaTolerance=getKeyVal(optx,["funcArg","zpk","deltaTolerance"],['yh_financials',{'ticker'},1800])
	modules = optx.pop('modules','')
	jobj = {colx:ticker}
	df=[]
	try:
		df=find_mdb(jobj,clientM=clientM,dbname=dbname,tablename=tablename,dfTF=True)[0]
		if len(df)>0 and isinstance(df,pd.DataFrame):
			sys.stderr.write(" --{} found in {}:\n{}\n".format(ticker,tablename,df)[:100]+"\n")
			return df
		else:
			datax=runOTF(funcArg,ticker,deltaTolerance=deltaTolerance,modules=modules,dbname=dbname,tablename=tablename,zpk=zpk)
			if len(datax)<1:
				sys.stderr.write(" --{} not found via {} @{}\n".format(ticker,modules,'tk2infoM'))
				return []
			df= pd.DataFrame(datax)
			sys.stderr.write(" --{} found in {}:\n{}\n".format(ticker,modules,df)[:100]+"\n")
			return df
	except:
		sys.stderr.write("**ERROR:{}: {}\n".format(ticker,str(e)))
	return df

def tk2info(ticker='',tablename='mapping_ticker_cik',colx='ticker',pgDB=None):
	if ticker is None or len(ticker)<1:
		return {}
	elif isinstance(ticker,list):
		ticker=ticker[0]
	xqTmp = """SELECT * FROM "{tablename}" WHERE {colx}='{ticker}'"""
	xqr = xqTmp.format(tablename=tablename,ticker=ticker,colx=colx)
	sys.stderr.write(" --find {} from SQL:\n{}\n".format(ticker,xqr))
	df= sqlQuery(xqr,engine=pgDB)
	if len(df)>0 and isinstance(df,pd.DataFrame):
		return df
	else:
		return []
       
def en2cn(e='Metropolitan Bank Holding Corp'):
	from googletrans import Translator
	c = Translator().translate(e,src='en',dest='zh-tw')
	sys.stderr.write(" --Translate: {} -> {}\n".format(e,c.text))
	return c.text

def batch_t2l(tkLst=[],output='dict',quoteTF=True,dbname='ara'):
	dd=[] if output=='dict' else pd.DataFrame()
	for ticker in tkLst:
		ret=t2l(ticker,output=output,quoteTF=quoteTF,dbname=dbname)
		if len(ret)<1:
			continue
		if isinstance(dd,pd.DataFrame):
			dd=dd.append(ret,ignore_index=True)
		elif isinstance(dd,list):
			dd.extend(ret)
		else:
			dd.append([ret])
	return dd
	
def ticker2label(ticker='',output='dict',quoteTF=True,dbname='ara'):
	return t2l(ticker=ticker,output=output,quoteTF=quoteTF,dbname=dbname)

def t2l(ticker='',output='dict',quoteTF=True,dbname='ara'):
	''' get basic info from ticker
	'''
	if isinstance(ticker,list):
		return batch_t2l(tkLst=ticker,output=output,quoteTF=quoteTF,dbname=dbname)

	#- GET summaryProfile from mDB:yh_summaryProfile or onTheFly
	dg = tk2infoM(ticker,tablename="yh_summaryProfile",funcArg='yh_financials',modules="summaryProfile",zpk={'ticker'},deltaTolerance=864000)
	dg = renameDict(dg,{"sector":"sector_alias"})

	#- GET basic ticker info from pgDB:mapping_ticker_cik 
	pgDB = conn2pgdb(dbname=dbname)
	df = tk2info(ticker,tablename='mapping_ticker_cik',pgDB=pgDB)
	if all([len(df),len(dg)]):
		df = df.merge(dg,on='ticker')
	elif len(dg)>0:
		df=dg

	#- CHECK available data ---------------------------#
	if len(df)<1:
		return {}

	#- GET ticker sector info from pgDB:spdr_sector 
	if 'sector_alias' in df:
		sa = 'sector_alias' 
		dg = tk2info(df[sa].values[0],tablename='spdr_sector',colx=sa,pgDB=pgDB)
	elif 'sector' in df:
		sa = 'sector' 
		dg = tk2info(df[sa].values[0],tablename='spdr_sector',colx=sa,pgDB=pgDB)
	else:
		dg=[]

	if all([len(df),len(dg)]):
		df = df.merge(dg[['etfname','sector','sector_alias','sector_cn']],on=sa)
		if "sector_cn" not in df:
			df = renameDict(df,{"sector_cn_x":"sector_cn","sector_x":"sector"})
	elif len(dg)>0:
		df=dg

	if 'industry' in df:
		dg = tk2info(df['industry'].values[0],tablename='sector_alias_industry',colx='industry',pgDB=pgDB)
		if len(dg):
			df = df.merge(dg[['industry','industry_cn']],on='industry')
		else:
			df['industry_cn'] = df['industry'].values

	#- GET summaryProfile from mDB:yh_quote_curr or onTheFly
	dg = tk2infoM(ticker,tablename='yh_quote_curr',funcArg='yh_quote_comparison',zpk={'ticker'},deltaTolerance=900)
	if all([len(df),len(dg)]):
		df = df.merge(dg,on='ticker')
		if 'trailingPE' in df:
			df['peRatio']=df['trailingPE']
	elif len(dg)>0:
		df=dg

	if 'sector' not in df and 'quoteType' in df:
		df['sector_cn'] = df['sector'] = df['quoteType']

	try:
		if 'shortName' in df:
			df['company'] = df['shortName']
		if not df['company'].values[0]:
			df['company'] = df['ticker']
		if 'company_cn' not in df:
			df['company_cn'] = en2cn(df['company'].values[0])
	except Exception as e:
			sys.stderr.write("**ERROR:{} via {} in {}\n".format(str(e),'en2cn', 't2l()'))
	if output=='dict':
		return df.to_dict(orient='records')[0]
	return df

if __name__ == '__main__':
	tkLst=sys.argv[1:]
	if tkLst:
		df = ticker2label(tkLst,output=None,quoteTF=True)
		if len(df)>0:
			sys.stderr.write("\n --OUT:\n{}\n".format(df.to_dict(orient='records')))
	else:
		sys.stderr.write("No ticker {} assigned.\n".format(tkLst))
