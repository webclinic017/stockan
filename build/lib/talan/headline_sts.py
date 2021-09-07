#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
'''
Find leading stocks (up/down > 1%) w.r.t. SP500/DowJone's up/down movement
preparation for generate daily headline context
Usage of
python3 headline_sts.py NDAYS > headline_hist.txt
OR To save result to mongoDB table ara::headline_hist
NDAYS=2; python3 headline_sts.py $NDAYS 1 > headline_hist.txt
OR To get headline_hist for end at specific date:'end'
python -c "from headline_sts import headline_hist as hh;print(hh(ndays=2,saveDB=False,end=20190607).to_csv(index=False))"

also see,
record_hilo.py
Note: python2/3 compatiable
'''
from _alan_calc import sqlQuery,conn2pgdb
from _alan_optparse import parse_opt, subDict
import sys
import ast
if sys.version_info.major == 2:
	from cStringIO import StringIO
else:
	from io import StringIO
import numpy as np
import pandas as pd

# use xqTmp, tkLst
def get_stock_hist(tkLst,ndays=2,xqTmp='',diffTF=False,pgDB=None,end=None,debugTF=False,liveTF=False):
	ticker=tkLst[0]
	xqx = "select pbdate from prc_hist where name='{}' {} order by pbdate DESC limit {}"
	addiClause = '' if end is None else ' AND pbdate<={}'.format(end)
	dtmp = sqlQuery(xqx.format(ticker,addiClause,ndays),pgDB)
	currdb_date = dtmp['pbdate'].iloc[0]
	xdb_date = dtmp['pbdate'].iloc[1]
	if end is not None and int(end)>currdb_date and liveTF is True:
		from yh_hist_batch import yh_hist
		from _alan_date import next_date
		s = next_date(end,dformat='%Y%m%d',days=-5,dtTF=False)
		ranged = "{},{}".format(s,end)
		dtmp = yh_hist(ticker,gap='1d',ranged=ranged,debugTF=debugTF)
		sys.stderr.write("===LIVE dtmp:\n{}\n".format(dtmp.tail()))
		curryh_date = dtmp['pbdate'].iloc[-1]
	else:
		curryh_date = currdb_date
	if curryh_date>currdb_date:
		dx=pd.DataFrame()
		for ticker in tkLst:
			ticker = ticker.replace('.','-')
			dtmp = yh_hist(ticker,gap='1d',ranged=ranged,debugTF=debugTF)[-ndays:]
			dtmp['chgPct'] = dtmp['close'].pct_change().values*100.
			dtmp['chgLevel'] = dtmp['close'].diff(periods=1).values
			dx= pd.concat([dx,dtmp])
		dx.rename(columns={'name':'ticker'},inplace=True)
		dx.reset_index(drop=True,inplace=True)
		sys.stderr.write("===LIVE dx:\n{}\n".format(dx.tail()))
		return dx
	else:
		start = xdb_date
	if len(xqTmp)<1:
		xqTmp = "select * from prc_hist WHERE name in {} and pbdate>={} {} ORDER BY name,pbdate"
	nmLst = [x.replace('.','-') for x in tkLst]
	nmPns="('{}')".format("','".join(nmLst))
	xqr = xqTmp.format(nmPns,start,addiClause)
	dx=sqlQuery(xqr,pgDB)
	dx['ticker']= [x.replace('-','.') for x in dx['name'] ]
	dx['chgPct']= np.empty(dx['ticker'].shape)
	if diffTF is True:
		dx['chgLevel']= np.empty(dx['ticker'].shape)
	for ticker in tkLst:
		dx.loc[dx['ticker']==ticker,'chgPct'] = dx.loc[dx['ticker']==ticker,'close'].pct_change().values*100.
		if diffTF is True:
			dx.loc[dx['ticker']==ticker,'chgLevel'] = dx.loc[dx['ticker']==ticker,'close'].diff(periods=1).values
	return dx

def alloc_yvalue3(vx,thd=[0,0],yvalue=[1,0,-1],openTF=True):
	'''
	Return elements from one of 'yvalue' depending on threshold condition: 'thd'
	'''
	if len(yvalue)<3:
		return vx
	if isinstance(vx,list):
		vx = np.array(vx)
	if hasattr(thd, "__len__"):
		if openTF is True:
			vs = np.where(vx>thd[0],yvalue[0],(np.where(vx<thd[1],yvalue[2],yvalue[1])) )
		else:
			vs = np.where(vx>=thd[0],yvalue[0],(np.where(vx<=thd[1],yvalue[2],yvalue[1])) )
	else:
		if openTF is True:
			vs = np.where(vx>thd,yvalue[0],(np.where(vx<thd,yvalue[2],yvalue[1])) )
		else:
			vs = np.where(vx>=thd,yvalue[0],yvalue[2] )
	return vs

def alloc_yvalue2(vx,thd=0,yvalue=[1,-1],openTF=True):
	'''
	Return elements from one of 'yvalue' depending on threshold condition: 'thd'
	'''
	if len(yvalue)<2:
		return vx
	if isinstance(vx,list):
		vx = np.array(vx)
	if openTF is True:
		vs = np.where(vx>thd,yvalue[0],yvalue[1])
	else:
		vs = np.where(vx>=thd,yvalue[0],yvalue[1])
	return vs

# find top-changes list
def find_top_changes(pbdate=20190405,dx={}):
	cLst = ['ticker','chgPct','pbdate','close']
	if 'chgLevel' in dx.columns:
		cLst = cLst + ['chgLevel']
	df = dx.query('pbdate=={}'.format(pbdate))[cLst]
	df.reset_index(drop=True,inplace=True)
	df['sign'] = np.where(df['chgPct']>0,1,(np.where(df['chgPct']<0,-1,0)) )
	df['UpDn'] = alloc_yvalue3(df['chgPct'],thd=[0.005,-0.005],yvalue=["UP","FLAT","DOWN"])
	caseUp = df.loc[np.where(df['chgPct']>=1)].sort_values(by='chgPct',ascending=False)
	caseDn = df.loc[np.where(df['chgPct']<=-1)].sort_values(by='chgPct',ascending=True)
	return df, caseUp, caseDn

def  get_eqtLst(minMarketCap=50000000000):
	'''
	return equity list for import highlite based on dow_component,sp500_component, yh_quote_curr
	and minimum size (100B) of marketCalp
	'''
	xqTmp = '''SELECT a.ticker FROM sp500_component a, yh_quote_curr b
		WHERE a.ticker=b.ticker AND
		(b."marketCap">{} OR a.ticker in (SELECT ticker FROM dow_component))'''
	xqr = xqTmp.format(minMarketCap)
	try:
		eqtLst =  list(sqlQuery(xqr)['ticker'])
	except Exception as e:
		eqtLst=['AAPL','ABT','ACN','ADBE','AMGN','AMZN','AVGO','AXP','BA','BAC','BRK-B','C','CAT','CMCSA','COST','CRM','CSCO','CVX','DHR','DIS','DOW','FB','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KO','LIN','LLY','LMT','MA','MCD','MDT','MMM','MRK','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM','PYPL','SBUX','T','TMO','TRV','TXN','UNH','UNP','UPS','UTX','V','VZ','WBA','WFC','WMT','XOM']
	return eqtLst

def headline_hist(ndays=2,saveDB=False,end=None,outTF=False,**optx):
	eqtLst = get_eqtLst()
	pgDB=conn2pgdb(dbname='ara')
	# get indices chgPct: idxRtn
	idxLst = ['^GSPC','^DJI','^IXIC','^SOX']
	idxRtnHist = get_stock_hist(idxLst,ndays=ndays,diffTF=True,pgDB=pgDB,end=end)
	if len(idxRtnHist)<1:
		return []

	# get equity chgPct
	eqtRtnHist = get_stock_hist(eqtLst,ndays=ndays,pgDB=pgDB,end=end)
	if len(eqtRtnHist)<1:
		return []

	# get equity chgPct
	pbLst = idxRtnHist.loc[idxRtnHist['ticker']==idxLst[0],'pbdate'].sort_values(ascending=False)[:]
	# redirect stdout
	old_stdout = sys.stdout
	mystdout = StringIO()
	sys.stdout = mystdout
	fwrite=sys.stdout.write

	fwrite("pbdate|Index1|chgPct1|chgLevel1|UpDn1|close1")
	fwrite("|Index2|chgPct2|chgLevel2|UpDn2|close2")
	fwrite("|Index3|chgPct3|chgLevel3|UpDn3|close3")
	fwrite("|Index4|chgPct4|chgLevel4|UpDn4|close4")
	print("|topDate1|topUD1|topLst1|topDate2|topUD2|topLst2|conflict")
	for pbdate in pbLst:
		idxRtn,idxUp,idxDn = find_top_changes(pbdate=pbdate,dx=idxRtnHist)
		eqtRtn,eqtUp,eqtDn = find_top_changes(pbdate=pbdate,dx=eqtRtnHist)
		#print(idxRtn,"\n",eqtUp,"\n",eqtDn)
		if len(idxRtn)<1: 
			sys.stderr.write("skip date: {}\n".format(pbdate))
			continue

		sp500Ret = idxRtn.loc[idxRtn['ticker']=='^GSPC'].iloc[0].to_dict()
		dowRet = idxRtn.loc[idxRtn['ticker']=='^DJI'].iloc[0].to_dict()
		nasdaqRet = idxRtn.loc[idxRtn['ticker']=='^IXIC'].iloc[0].to_dict()
		phlRet = idxRtn.loc[idxRtn['ticker']=='^SOX'].iloc[0].to_dict()
		if np.isnan(sp500Ret['chgPct']):
			sys.stderr.write("skip date: {}\n".format(pbdate))
			continue

		conflict = 0 if dowRet['sign'] == sp500Ret['sign'] else 1

		eqtLx1 = eqtUp if sp500Ret['sign']==1 else eqtDn
		if len(eqtLx1)<1:
			topLst1=[]
			eqtLead1 = {}
		else:
			dtmp=eqtLx1[['ticker','chgPct']].head(3).to_dict(orient='records')
			topLst1=[{x['ticker']:round(x['chgPct'],2)} for x in dtmp]
			#topLst1=["{!r}:{:.2f}".format(*x) for x in (eqtLx1[['ticker','chgPct']].head(3).values)]
			eqtLead1 = eqtLx1.iloc[0].to_dict()
			eqtLead1['topLst1']=topLst1

		eqtLx2 = eqtUp if dowRet['sign']==1 else eqtDn
		if len(eqtLx2)<1:
			topLst2=[]
			eqtLead2 = {}
		else:
			dtmp=eqtLx2[['ticker','chgPct']].head(3).to_dict(orient='records')
			topLst2=[{x['ticker']:round(x['chgPct'],2)} for x in dtmp]
			#topLst2=["{!r}:{:.2f}".format(*x) for x in (eqtLx2[['ticker','chgPct']].head(3).values)]
			eqtLead2 = eqtLx2.iloc[0].to_dict()
			eqtLead2['topLst2']=topLst2

		fwrite("{pbdate}|{ticker}|{chgPct:.2f}|{chgLevel:.0f}|{UpDn}|{close:5g}".format(**sp500Ret))
		fwrite("|{ticker}|{chgPct:.2f}|{chgLevel:.0f}|{UpDn}|{close:5g}".format(**dowRet))
		fwrite("|{ticker}|{chgPct:.2f}|{chgLevel:.0f}|{UpDn}|{close:5g}".format(**nasdaqRet))
		fwrite("|{ticker}|{chgPct:.2f}|{chgLevel:.0f}|{UpDn}|{close:5g}".format(**phlRet))
		if len(eqtLx1)>0:
			fwrite("|{pbdate}|{UpDn}|{topLst1}".format(**eqtLead1))
		if len(eqtLx2)>0:
			fwrite("|{pbdate}|{UpDn}|{topLst2}".format(**eqtLead2))
		fwrite("|{conflict}\n".format(conflict=conflict))

	# redirect the stdout to string and convert it to dataframe
	xstr = mystdout.getvalue()
	sys.stdout = old_stdout
	fwrite=sys.stdout.write
	if outTF:
		fwrite(xstr)
	df = pd.read_csv(StringIO(xstr),sep='|')
	from _alan_calc import save2pgdb
	from _alan_str import write2mdb
	if saveDB==True:
		tablename="headline_hist"
		sys.stderr.write("Save to {}\n".format(tablename))
		#save2pgdb(df,db=pgDB,tablename=tablename)
		clientM=None
		zpk={"ticker","pbdate"}
		mobj,clientM,msg = write2mdb(df,clientM,tablename=tablename,zpk=zpk)

	if 'topDict' in optx and optx['topDict']==True:
		if 'topLst1' in df:
			df['topLst1']=[ ast.literal_eval(x) if hasattr(df['topLst1'], "__len__") else {} for x in df['topLst1'] ]
		if 'topLst2' in df:
			df['topLst2']=[ ast.literal_eval(x) if hasattr(df['topLst2'], "__len__") else {} for x in df['topLst2'] ]

	df['chg1']=df['chgLevel1']
	df['chg2']=df['chgLevel2']
	df['allUpDn']=0
	for j in range(df.shape[0]):
		if all([ x>0 for x in df[['chgPct1','chgPct2','chgPct3']].iloc[j] ]) :
			allUpDn = 1
		elif all([ x<0 for x in df[['chgPct1','chgPct2','chgPct3']].iloc[j] ]) :
			allUpDn = -1
		else:
			allUpDn = 0
		df.loc[df.index[j],'allUpDn']=allUpDn
	return df

if __name__ == '__main__':
	description="""Find leading stocks (up/down > 1%) w.r.t. SP500/DowJone's up/down movement"""
	(opts, args)=parse_opt(sys.argv,ndays=2,outTF=True,saveDB=False)
	df = headline_hist(topDict=True,**opts)
	print(df.to_dict(orient='records'))
