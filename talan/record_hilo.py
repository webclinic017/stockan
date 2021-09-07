#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
find record high/low returns 
jinjia2 script: templates/headtitle3_cn.j2
Usage of
python record_hilo.py ^DJI
OR 
python record_hilo.py ^GSPC
OR to save result to mongoDB table ara::record_hilo
python record_hilo.py ^GSPC --extra_xs=nlookback=360 --table=record_hilo
OR
record_hilo.py --extra_xs='nlookback=200' 'CL=F'
OR
record_hilo.py --extra_xs='nlookback=200' 'GC=F'
OR
record_hilo.py --extra_xs='nlookback=200' 'CNY=X'
OR
record_hilo.py --extra_xs='nlookback=200;colx="close";"' 'CNY=X'
OR
record_hilo.py --extra_xs='nlookback=200;colx="close";";mthLst=[-1,-3]' 'CNY=X'
'''

import sys
import numpy as np
from _alan_calc import pull_stock_data as psd,getKeyVal,conn2mgdb
from _alan_str import write2mdb,upsert_mdb
from _alan_date import next_date,delta2dates
from _alan_optparse import parse_opt, subDict
if sys.version_info.major == 2:
	reload(sys)
	sys.setdefaultencoding('utf8')

def is_record_hl(dx,colx='pchg',**nd_args):
	if len(nd_args)<1:
		startDT=next_date(dx.index[-1],month=1,day=1)
	else:
		startDT=next_date(dx.index[-1],**nd_args)
	ds = dx.loc[dx.index>=startDT,colx]
	mnxTF,hilo,idxmnx = find_hilo(ds)
	startAT=ds.index[0] # actual startDate in datetime
	#sys.stderr.write("===Act:{}, Start:{}, Args:{}\n".format(startAT,startDT,nd_args))
	return mnxTF,hilo,idxmnx,startDT,startAT

def find_hilo(ds):
	'''
	find hilo of pandas.Series 'ds'
	'''
	if len(ds)<1:
		return False,ds.iloc[-1]>0,None
	idxmnx=ds.index[-1]
	if ds.idxmax()==idxmnx:
		hilo="HIGH"
		mnxTF = True
	elif ds.idxmin()==idxmnx:
		hilo="LOW"
		mnxTF = True
	else:
		mnxTF = False
		hilo=""
	return mnxTF,hilo,idxmnx

def find_hiloOLD(ds):
	'''
	find hilo of pandas.Series 'ds'
	'''
	if len(ds)<1:
		return False,ds.iloc[-1]>0,None
	if ds.iloc[-1]>0:
		idxmnx = ds.idxmax()
		hilo="HIGH"
	else:
		idxmnx = ds.idxmin()
		hilo="LOW"
	mnxTF = idxmnx==ds.index[-1]
	return mnxTF,hilo,idxmnx

def recordHiLo(dx,endDT=None,ticker='',colx='pchg',colLst=['close','pchg'],debugTF=False,mthLst=[]):
	'''
	find hi/lo out of a range past months: 'mthLst':[-1,-3,-6,-12,-36,-60,-120]
	based on time series DataFrame 'dx' that contains column 'colx' with DatetimeIndex
	return dict object with keys:
	  fqWd, name, tags, days, months, years, endDT, mnxTF, hilo, startDT and 'colLst'
	'''
	if endDT is None:
		endDT=dx.index[-1] # next_date()
	if len(mthLst)<1:
		mthLst=[-1,-3,-6,-12,-36,-60,-120]
	pbdate = int(endDT.strftime('%Y%m%d'))
	kyLst = ["fqWd","name","tags","days","months","years","endDT","hilo","startDT"]
	if debugTF:
		sys.stderr.write("{}\n".format(dx))
	da=db={}
	jobj=dict(ticker=ticker,pbdate=pbdate,YTD={})
	for months in mthLst:
	#for months in [-6,-12,-36,-60,-120]:
		mnxTF,hilo,idxmnx,startDT,startAT = is_record_hl(dx,colx=colx,months=months)
		ndays = (endDT-startAT).days
		nmonths=delta2dates(endDT,startAT,fq='M',rounding=1)
		nyears=int(nmonths/12) # only integer year
		startYmd=int(startAT.strftime('%Y%m%d'))
		if delta2dates(startAT,startDT)>5: # STOP if act-delta-months less than months
			break 
			name = "{}_MoSinceStart".format(nmonths)
		else:
			#name = "{}_Mo2Date".format(-months)
			name = "YrToDate" if nyears>0 else "LastMoToDate"

		if (mnxTF is True):
			colVa = dx.loc[idxmnx,colx]
			hlWd  = '最高' if hilo=='HIGH' else '最低' if colVa>=0 else '最差'
			colWd = '價格' if colx in ['price','close','value'] else '變化'
			if nyears<1:
				tags = "{} month {} {}".format(-months,colx,hilo)
				fqWd = "{} 個月以來 {} {}".format(-months,hlWd,colWd)
			else:
				tags = "{} year {} {}".format(nyears,colx,hilo)
				fqWd = "{} 年以來 {} {}".format(nyears,hlWd,colWd)
			vaLst = [fqWd,name,tags,ndays,nmonths,nyears,pbdate,hilo,startYmd]
			db = dict(zip(kyLst,vaLst))
			db.update(dx.loc[idxmnx,colLst].to_dict())
			jobj.update(YTD=db)
			if delta2dates(startAT,startDT)>5: #- No longer used since longer delta months is not applicable
				sys.stderr.write("===Act:{}, Start:{}, Months:{}\n".format(startAT,startDT,months))
				break
	return jobj

def recordHiLo2(dx,endDT=None,ticker='',colx='pchg',colLst=['close','pchg'],debugTF=False,mthLst=[],colx2='close'):
	'''
	find hi/lo out of a range past months: 'mthLst':[-1,-3,-6,-12,-36,-60,-120]
	based on time series DataFrame 'dx' that contains columns 'colx','colx2' with DatetimeIndex
	and find hi/lo based on 'colx2' 1st then 'colx' if 1st try is an empty {}.
	return dict object with keys:
	  fqWd, name, tags, days, months, years, endDT, mnxTF, hilo, startDT and 'colLst'
	'''
	jobj = recordHiLo(dx,endDT=endDT,ticker=ticker,colx=colx,colLst=colLst,mthLst=mthLst)
	if len(jobj['YTD'])<1:
		jobj = recordHiLo(dx,endDT=endDT,ticker=ticker,colx=colx2,colLst=colLst,mthLst=mthLst)
	return jobj

# DEPRECATED
def find_record_hilo(dx,endDT=None,ticker='',colx='pchg',colLst=['close','pchg'],debugTF=False,mthLst=[]):
	if endDT is None:
		endDT=next_date()
	pbdate = int(endDT.strftime('%Y%m%d'))
	kyLst = ["name","days","endDT","mnxTF","hilo","startDT"]
	if debugTF:
		sys.stderr.write("{}\n".format(dx))
	da=db={}
	jobj=dict(ticker=ticker,pbdate=pbdate,MTD={},YTD={})
	if endDT.day<20: # Last MoToDate
		mnxTF,hilo,idxmnx,startDT,startAT = is_record_hl(dx,colx=colx,months=-1,day=1)
		ndays = (endDT-startDT).days
		startYmd=int(startDT.strftime('%Y%m%d'))
		name = "LastMoToDate"
		if (mnxTF is True):
			vaLst = [name,ndays,pbdate,mnxTF,hilo,startYmd]
			da = dict(zip(kyLst,vaLst))
			da.update(dx.loc[idxmnx,colLst].to_dict())
			jobj.update(MTD=da)
	else: # MoToDate
		mnxTF,hilo,idxmnx,startDT,startAT = is_record_hl(dx,colx=colx,day=1)
		ndays = (endDT-startDT).days
		startYmd=int(startDT.strftime('%Y%m%d'))
		name = "MoToDate"
		if (mnxTF is True):
			vaLst = [name,ndays,pbdate,mnxTF,hilo,startYmd]
			da = dict(zip(kyLst,vaLst))
			da.update(dx.loc[idxmnx,colLst].to_dict())
			jobj.update(MTD=da)
			
	# YrToDate
	mnxTF,hilo,idxmnx,startDT,startAT = is_record_hl(dx,colx=colx,month=1,day=1)
	ndays = (endDT-startDT).days
	startYmd=int(startDT.strftime('%Y%m%d'))
	name = "YrToDate"
	if (mnxTF is True) and ndays>45:
		vaLst = [name,ndays,pbdate,mnxTF,hilo,startYmd]
		db = dict(zip(kyLst,vaLst))
		db.update(dx.loc[idxmnx,colLst].to_dict())
		jobj.update(YTD=db)
	return jobj

def record_hilo_tst(opts={},**optx):
	## ASSIGN local variables
	if len(opts)<1:
		opts, args= parse_opt(sys.argv)
	opts.update(optx)
	ticker = args[0] if len(args[0])>0 else '^GSPC'
	ticker = getKeyVal(opts,'ticker',ticker)
	debugTF = getKeyVal(opts,'debugTF',False)
	start,end,days = getKeyVal(opts,['start','end','days'],[None,None,3600])
	tablename = getKeyVal(opts,'tablename',None)
	funcName = getKeyVal(opts,'funcName','recordHiLo2')
	nlookback = getKeyVal(opts,'nlookback',1)
	src = getKeyVal(opts,'src','yh')
	searchDB = getKeyVal(opts,'searchDB',True)
	colLst = getKeyVal(opts,'colLst',['close','pchg'])
	colx = getKeyVal(opts,'colx','pchg')
	mthLst = getKeyVal(opts,'mthLst',[])
	if debugTF==True:
		sys.stderr.write("OPTS:{}\n".format(opts))
	## ARRANGE additional local variables
	nlookback = -int(nlookback)
	if funcName in globals():
		funcArg = globals()[funcName]
	else:
		funcArg = recordHiLo
	clientM = None
	## GET DATA
	df = psd(ticker,start=start,end=end,days=days,src=src,searchDB=searchDB,debugTF=debugTF)
	if debugTF==True:
		sys.stderr.write("DF:\n{}\n".format(df.tail()))
	if colx=='pchg' and colx not in df and 'close' in df:
		df['pchg'] = np.round(df['close'].pct_change(),4)
	## LOOPING funcArg for backtest
	for xd in df.index[nlookback:] :
		try:
			dx = df.loc[df.index<=xd]
			pbdate = int(xd.strftime('%Y%m%d'))
			jobj = funcArg(dx,endDT=xd,ticker=ticker,colx=colx,colLst=colLst,mthLst=mthLst)
			if debugTF==True:
				sys.stderr.write("{}\n".format(dx.iloc[-1]))
			#if any([jobj['MTD'],jobj['YTD']]):
			if jobj['YTD']:
				sys.stderr.write("{}\n".format(jobj['YTD']))
				if tablename is not None:
					zpk={"ticker","pbdate"}
					mobj,clientM,msg = write2mdb(jobj,clientM,tablename=tablename,zpk=zpk)
		except Exception as e:
			continue
	return jobj

def titlehead_tst(opts={},**optx):
	ds = titlehead_backtest(opts,**optx)
	sys.stderr.write( "ds:\n{}\n".format(ds.to_string()))
	return ds

def get_titlehead(opts={},**optx):
	optx.update(nlookback=1)
	ds = titlehead_backtest(opts,**optx)
	if len(ds)<1:
		return {}
	ret = dict(ht3=ds.iloc[0]['comment'],f=ds)
	sys.stderr.write( "ht3:{}\nf:\n{}\n".format(ret['ht3'],ret['f'].to_string()))
	return ret
	
def titlehead_backtest(opts={},**optx):
	import pandas as pd
	from _alan_calc import sqlQuery, pull_stock_data as psd
	from _alan_str import jj_fmt
	dirname,lang,mp3YN = getKeyVal(optx,['dirname','lang','mp3YN'],['templates','cn',False])
	nlookback = getKeyVal(optx,'nlookback',1)
	days = getKeyVal(optx,'days',3700)
	searchDB = getKeyVal(optx,'searchDB',True)
	debugTF = getKeyVal(optx,'debugTF',False)
	dbname = optx.pop('dbname','ara')
	tablename = optx.pop('tablename',None)
	dLst=optx.pop('args',None)
	if dLst is None or len(dLst)<1:
		dLst = sqlQuery("SELECT * FROM mapping_series_label WHERE freq='D' and category_label_seq>0")
	else:
		dLst = sqlQuery("SELECT * FROM mapping_series_label WHERE freq='D' and category_label_seq>0 and series in ('{}')".format("','".join(dLst)))
	ds=[]
	for lkbx in range(nlookback):
		dm=[]
		for jx in range(len(dLst)):
			ticker,freq,src,label_cn,category_cn = dLst[['series','freq','source','label_cn','category_cn']].iloc[jx]
			if freq != 'D':
				continue
			df = psd(ticker,days=days,src=src,debugTF=debugTF,pchgTF=True,searchDB=searchDB)
			try:
				dx = df.iloc[:-lkbx] if lkbx>0 else df
				ret=recordHiLo2(dx,ticker=ticker)
				if len(ret['YTD'])>0:
					dd=dLst.iloc[jx].to_dict()
					dd.update(ret['YTD'])
					cmt=jj_fmt('headtitle3_cn.j2',lang=lang,mp3YN=False,dirname=dirname,ctrlSTRUCT='include',**dd)
					mp3cmt=jj_fmt('headtitle3_cn.j2',lang=lang,mp3YN=True,dirname=dirname,ctrlSTRUCT='include',**dd)
					dd.pop('_id',None)
					dd['ticker'] = ticker
					dd['pbdate'] = dd['endDT']
					dd['comment'] = cmt
					dd['mp3comment'] = mp3cmt
					emsg="{}\n".format((jx,category_cn,freq,ticker,src,label_cn,dd['fqWd'],dd['endDT']))
					sys.stderr.write(emsg)

					dm.append(dd)
					if debugTF:
						sys.stderr.write("RUNNING {}:{}:{}\n{}\n".format(lkbx,jx,ticker,ret))
			except Exception as e:
				sys.stderr.write("**ERROR:{}:{}:{}\n".format(jx,ticker,str(e)))
				continue
		if len(dm)<1:	
			continue
		dm = pd.DataFrame(dm)
		ds = dm.sort_values(by=['days','category_seq','category_label_seq'],ascending =[False,True,True])
		if tablename is not None:
			zpk={"ticker","pbdate"}
			sys.stderr.write("Save to {}:\n{}\n".format(tablename,dm))
			mobj,clientM,msg = upsert_mdb(dm,tablename=tablename,dbname=dbname,zpk=zpk)
	return ds

if __name__ == '__main__':
	description="""Find record high/low returns, e.g., record_hilo.py SPY --extra_xs=nlookback=100"""
	fx = lambda ky,va,m: m[ky] if ky in m and m[ky] in globals() else va
	opts, args = parse_opt(sys.argv, description=description)
	#funcName=opts['main'] if 'main' in opts and opts['main'] in globals() else 'get_titlehead'
	funcName=fx('main','get_titlehead',opts)
	funcArg = globals()[funcName]
	sys.stderr.write("===RUN:{}:({})\n".format(funcName,opts))
	df = funcArg(**opts)
