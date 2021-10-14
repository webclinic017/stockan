#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
program: _alan_str.py
Description: ALAN string functions for NLG 
Version: 0.80
Functions:
  remove_tags(raw_html):
  unique_set(a):
  check_latest_macro(cdt=None,nwk=1):
  dLst2df(dx,keyName='ticker',valName='pctChange'):
  merge_t2l(tkLst=[],df={},colx='ticker',**optx):
  merge_yqc(tkLst=[],df={},colx='ticker',**optx):
  merge_stock_info(tkLst=[],df={},colx='ticker',how='left',**optx):
  hourly_eod(ticker,start=None,mp3YN=False,src='yh',lang='cn',end_hm=1600,dirname='templates/',outdir='US/mp3_hourly',**optx):
  syscall_eod(ticker,start=None,mp3YN=False,src='yh',wkDir='.'):
  find_numbers(s,convertTF=False,pattern='',prefix='',suffix=''):
  sysCall(xcmd=None):
  popenCall(xcmd=None,xin=None,bufsize=1024,shell=True,debugTF=False):
  str2float(s,r=None):
  strc2float(s,r=None,sc=','):
  str2value(s,r=None):
  num_en2cn(s='10B',numTF=False):
  sub_special_char(s='',p='{}[]()#%|*,:;/ ',n=''):
  consecutive_sign(v,zeroSign=0,reverseTF=True):
  run_jj2(ts,dd={},dirname='.',fileTF=False,debugTF=False):
  jj2_fmt(ts,dirname='.',fileTF=False,debugTF=False,j2onlyTF=False,**dd):
  get_arg2func(funcName,**optx):
  get_funcLst(unLst = ['sysCall','popenCall','udfStr_DEPRECATED']):
  jj_fmt(ts,dd={},dirname='.',fileTF=False,debugTF=False,j2onlyTF=False,**opts):
  ifelse(yn,a,b):
  clip(a,x,b):
  isclip(a,x,b):
  isclipL(a,x,b):
  isclipR(a,x,b):
  num2MP3(s,dotSign='點',lang="cn",mp3YN=True):
  gsrg(r,n):
  roundTBM(r,n):
  roundSD(r,n,lang='cn',mp3YN=False,TBMTF=False): 
  roundPctMP3(r,n=0,pct=100,dotSign='.',lang='en'):
  roundPct(r,n=0,pct=100,dotSign='.',lang='en',mp3YN=False):
  loc_dindex(r,thd=[1.96,1.15,0.32,0.1,-0.1,-0.32,-1.15,-1.96],openTF=True,sortTF=False):
  loc_aindex(r,thd=[-1.96,-1.15,-0.32,-0.1,0.1,0.32,1.15,1.96],openTF=True,sortTF=False):
  udfWord(r,wordLst=None,thd=None,lang='cn',openTF=True,sortTF=False,ascendingTF=False,scale=1):
  performance_verb(r,udfLst):
  btwDates(a,b,bracket=None,fmt=None,lang="en"):
  btwDates_cn(a,b,bracket=None,fmt=None):
  btwDates_en(a,b,bracket=None,fmt=None):
  datetime2hm(dtm,dformat="%H:%M",lang="cn",mp3YN=False):
  ymd2datetime(s,dformat="%Y%m%d"):
  ymd2ym(s,ym=None,ymd='%Y-%m-%d',lang="cn"):
  ymd2ymd(s,ym=None,ymd='%Y-%m-%d',lang="cn",):
  ymd2md(s,ym='%m-%d',ymd='%Y%m%d',lang="en",mp3YN=False):
  pattern_cn(name,lang="cn"):
  fq2unit_str(fqStr,lang="cn"):
  rdWord(wordLst,sep="|"):
  udfSet(udf=None,lang="en",tense="past"):
  udfStr(r,udf=None,zs=[0,0],lang="en",tense="past"):
  udfStr_DEPRECATED(r,udf=None,zs=0,lang="en",tense="past"):
  roundMP3(x,n=0,usdSign=None,dotSign='.',lang=None):
  roundAuto(x,n=2,usdSign=None,dotSign='.',lang='en',mp3YN=False):
  roundUSD(x,n=0,usdSign='$',dotSign='.',lang='en',mp3YN=False):
  wrapWord(a,bk=["(",")"]):
  rangeStr(a,b,bk=["[","]"]):
  btwStr(a,b,bk=["ranging between","and"],fl="fluctuated around"):
  btwStr_cn(a,b,bk=["介于","至","的範圍之間"],fl=["變化不大，只會在","內上下稍稍波動"],unitStr=''):
  connMdb(clientM=None):
  find_mdb(jobj={},clientM=None,dbname='ara',tablename=None,field={},sortLst=[],ascendingTF=False,limit=0,dfTF=False,**optx):
  get2mndb(jobj,tablename='lsi2nlg',zpk={'ticker','username','tmplname','category','tmpltype','lang'}):
  upsert_mdb(jobj={},clientM=None,port=27017,dbname='ara',tablename=None,wmode='upsert',zpk=[],zasc=[],**optx):
  write_mdb(jobj={},clientM=None,port=27017,dbname='ara',tablename=None,wmode='upsert',zpk=[],zasc=[],**optx):
  insert_mdb(jobj={},clientM=None,port=27017,dbname='ara',tablename=None,wmode='',zpk=[],zasc=[],**optx):
  upd2mdb(dd=[],zpk={},mcur={},debugTF=False):
  np2mongo(zobj):
  write2mdb(jobj,clientM=None,dbname='ara',tablename=None,zpk={},insertOnly=False):
  save2mgdb(mLst=None,mky=None,dbM=None,tablename="temp",wmode='replace'):
  df_tofile(fp,df,output=None,sep='|',indexTF=False,hdrTF=True):
  str_tofile(fp,s):
  df_output(df,output=None,sep='|',indexTF=False,hdrTF=True):
  save2mndb(jobj,zpk={'ticker','username','tmplname','category','tmpltype','lang'},clientM=None,dbname='ara'):
  lst2dict(jkst,jlst):
  tmpl2lsi(tmplstr,argobj):
  combine_cmd2dict(*_xargs_): # TBD ,**_xdu_):
  jobj2lsi(jobj):
  lsi2nlg_calc(jobj):
  _alan_nlg(debugTF=False):
  qs_split(xstr,d1='&',d2='='):
  qs_exec(xstr):
  qs_execOLD(xstr):
  extra_opts(opts={},xkey='',method='JS',updTF=True):
  generate_comment_hourly(ts,df,opts=None):
  trend_forecast(ticker='^GSPC',freq='W',debugTF=False,pgDB=None,**optx):
  stock_screener(scrIds='most_actives',sortLst=['dollarValue'],ascLst=[False],**optx):
  x_screener(nobs=6,sortLst=['dollarValue'],ascLst=[False],dbname='ara',tablename='yh_qutoe_curr',addiFilter=1,debugTF=False,colx=[],addiClause='',**optx):
Last Mod., 
Wed May 27 13:45:46 EDT 2020
Tue Aug  3 16:25:32 EDT 2021
"""

#from __future__ import print_function
import sys
import os
from jinja2 import Environment as jinEnv, FileSystemLoader
import jinja2.ext
import pickle
from pymongo import MongoClient
from bson.binary import Binary
from string import Formatter
from math import *
import random
import locale
import datetime
import types
import json
import numpy as np
import pandas as pd
from pandas import Timestamp
import re
from subprocess import Popen, PIPE, STDOUT
from importlib import import_module
# IMPORT external functions
from _alan_date import next_date,ymd2dt,dt2ymd,s2dt
from _alan_calc import saferun,subDict,subDF,subVDict,renameDict,getKeyVal,sqlQuery,conn2mgdb,saferun

if sys.version_info.major == 2:
	reload(sys)
	sys.setdefaultencoding('utf8')

def str_contains(va=[],subs=[],vb=[]):
	'''
	search string list:va that contains any substrings:subs and add addtional list vb
	and return a unique list of strings
	'''
	return list(set([x for x in va if any(map(x.__contains__,subs))]+vb))

def clean_punctuation2upper(s,c=' '):
	from string import punctuation
	xs = s.translate(s.maketrans(punctuation,c*32))
	xs = re.sub('\s+',' ',xs).strip().upper()
	return xs

def remove_tags(raw_html):
	''' remove HTML tags
	'''
	cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
	cleantext = cleanr.sub('', raw_html)
	#cleantext = re.sub(cleanr, '', raw_html)
	return cleantext

def unique_set(a):
	''' return unique set with the same order
	'''
	return sorted(set(a), key=a.index)

def check_latest_macro(cdt=None,nwk=1):
	'''
	return macro info in dict via checking 
	if latest macro event is within last 2 weeks since last Monday
	otherwise return None
	Note, 
	  If annualRate is calc when 'deriv' is added in 'mapping_series_label' source column 
	  'annualRate' will be added and pctChange will apply to 'annualRate' variable
	  rather than 'value'
	'''
	from _alan_date import tg_latest2week,ymd2dt
	xqr="""SELECT  x.value,x.pbdate,y.* from macro_hist_fred x, (SELECT a.vntdate,b.* FROM macro_vintage_date a, mapping_series_label b
where a.series=b.series and b.freq in ('M','Q') ORDER BY vntdate DESC limit 1) as y
WHERE x.series=y.series ORDER BY pbdate DESC limit 13 """
	#df = sqlQuery(open('latest_macro_vintage.sql').read())
	df = sqlQuery(xqr)
	in2wTF = tg_latest2week(ymd2dt(df['vntdate'].iloc[0]),cdt=cdt,nwk=nwk)

	#=  Check If annualRate is calc when source='deriv' in the 'mapping_series_label' table
	dg=sqlQuery("SELECT * from macro_hist_fred where series='{}' ORDER BY pbdate".format(df['series'].iloc[-1]+"_PCTCHG"))
	if len(dg)>0:
		dg.rename(columns={"value":"annualRate"},inplace=True)
		df = df.merge(dg[['annualRate','pbdate']],on='pbdate')
		df['pctChange'] = df['annualRate'].diff(-1)
	else:
		df['pctChange'] = df['value'].pct_change(-1)

	if in2wTF is not None and len(in2wTF)>0:
		in2wTF.update(df.iloc[0].to_dict())
		in2wTF.update(f=df)
	return in2wTF


def dLst2df(dx,keyName='ticker',valName='pctChange'):
	'''
	convert list of dictionay to dataframe
	'''
	if len(dx)<1:
		return {}
	df=pd.DataFrame([list(x.items())[0] for x in dx],columns=[keyName,valName]).set_index(keyName,drop=True)
	return df

def merge_t2l(tkLst=[],df={},colx='ticker',**optx):
	'''
	Merge ticker2label info into dataFrame 'df' based on 'colx' column
	where 'colx' default to 'ticker'
	'''
	from ticker2label import ticker2label
	kwargs=dict(output=None,funcArg=ticker2label)
	kwargs.update(optx)
	dm = merge_stock_info(tkLst,df,**kwargs)
	return dm

def merge_yqc(tkLst=[],df={},colx='ticker',**optx):
	'''
	Merge stock quote info into dataFrame 'df' based on 'colx' column
	where 'colx' default to 'ticker'
	'''
	from yh_chart import yh_quote_comparison
	kwargs=dict(screenerTF=False,funcArg=yh_quote_comparison)
	kwargs.update(optx)
	dm = merge_stock_info(tkLst,df,**kwargs)
	return dm

def merge_stock_info(tkLst=[],df={},colx='ticker',how='left',**optx):
	from ticker2label import ticker2label
	funcArg=getKeyVal(optx,'funcArg',ticker2label)
	kwargs={}
	kwargs.update(optx)
	kwargs.pop('funcArg',None)
	dq = funcArg(tkLst,**kwargs)
	if len(dq)<1 or colx not in dq:
		return df
	if colx not in df:
		return dq
	return df.merge(dq,on=colx,how=how)

def jsonCvt(x):
	from bson.objectid import ObjectId
	from pandas import Timestamp
	import datetime
	if isinstance(x, (ObjectId,datetime.datetime,Timestamp)):
		return x.__str__()
	elif isinstance(x, np.ndarray):
		return list(x)
	elif isinstance(x, np.integer):
		return int(x)
	elif isinstance(x, np.float):
		return float(x)
	return x

def jsonObjReload(zobj,cvtFunc=jsonCvt):
	if cvtFunc is None:
		return zobj
	return json.loads(json.dumps(zobj,default=cvtFunc))

def hourly_eod(ticker,start=None,mp3YN=False,src='yh',lang='cn',end_hm=1600,dirname='templates/',outdir='US/mp3_hourly',**optx):
	'''
	hourly EoD run
	'''
	from ticker2label import ticker2label
	from hourly_mkt import wrap_hourly 
	tkInfo = ticker2label(ticker)
	if len(tkInfo)>0:
		optx.update(tkInfo)
	cname = 'company_cn' if lang == 'cn' else 'company'
	title=optx[cname] if cname in tkInfo  else ticker
	optx.update(start=start,mp3YN=mp3YN,src=src,lang=lang,dirname=dirname,title=title,outdir=outdir)
	end_hm= clip(900,end_hm,1600)
	sys.stderr.write("=====Running {}:{} @{}\n".format(start,end_hm,"hourly_eod"))
	jobj = wrap_hourly([ticker],end_hm=end_hm,**optx)
	if 'hourlyObj' in optx and isinstance(optx['hourlyObj'],dict):
		# Static setup fields
		fields=['ticker','comment','mp3comment','change','changePercent','copen','close','cprice','xprice','cpchg','marketCap','sector','title','rpt_hm','rpt_time','f']
		# Dynamic setup fields
		#fields = [ky for ky,val in jobj.items() if isinstance(val,(str,int,float)) ]
		#fields.append('f')
		fieldTyp = {ky:type(val) for ky,val in jobj.items() if ky in fields}
		sys.stderr.write(" --field/type saved for hourlyObj:{}\n".format(fieldTyp))
		optx['hourlyObj'].update(subDict(jobj,fields))
	return jobj['mp3comment'] if mp3YN else jobj['comment']

# DEPREATED, use hourly_eod
def syscall_eod(ticker,start=None,mp3YN=False,src='yh',wkDir='.'):
	'''
	shortcut systemCall for EoD run
	'''
	from ticker2label import ticker2label
	hourlyTmp='(cd {wkDir}; hourly_mkt.py  --start={start} --extra_xs="archiveTest=True;mp3YN={mp3YN};mp3TF=False;target_hm=[1600];mp3TF=False;plotTF=False;outTF=True" {ticker} --src={src} --title="{company_cn}" )'
	tkInfo = ticker2label(ticker)
	if len(tkInfo)<1:
		tkInfo=dict(company_cn=ticker,ticker=ticker)
	xstr = hourlyTmp.format(wkDir=wkDir,start=start,mp3YN=mp3YN,src=src,**tkInfo)
	o,e = popenCall(xstr)
	sys.stderr.write("{} {}\n".format(o,e))
	return o

def find_numbers(s,convertTF=False,pattern='',prefix='',suffix=''):
	'''
	extract all the numbers contained in a string
	and convert them into float if 'convertTF' is True
	'''
	if not pattern:
		pattern="[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
	if prefix:
		pattern=prefix+pattern
	if suffix:
		pattern=pattern+suffix
	rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
	for j,x in enumerate(rr):
		if ',' in x :
			rr[j] = x.replace(',','')
		elif prefix and prefix in x:
			rr[j] = x.replace(prefix,'')
		elif suffix and suffix in x:
			rr[j] = x.replace(suffix,'')
	if convertTF:
		rr = [ float(x) for x in rr]
	return rr

def popen(args,s=None,shell=True,bufsize=1024,stdin=PIPE,stdout=PIPE,stderr=PIPE,debugTF=False,**optx):
	'''
	System call via subprocess.Popen
	Where
	  args: can be a string of shell command is shell=True or
		a list of command args to form shell comman
	  s:    is the input string for stdin if there is any
	'''
	if debugTF:
		sys.stderr.write("==SHELL_CMD:\n{}\n".format(args))
	p = Popen(args,shell=shell,bufsize=bufsize,stdin=stdin,stdout=stdout,stderr=stderr)
	xin = (s+"\n").encode('utf-8') if s else None
	o,e = p.communicate(input=xin)
	if debugTF==2:
		msgerr="===SHELL_CMD:\n{}\n---OUT:\n{}\n---ERR:\n{}\n".format(args,o.decode(),e.decode())
		sys.stderr.write(msgerr)
	return o,e

def sysCall(xcmd=None):
	""" run [xcmd] as system call  """
	return Popen(xcmd, shell=True, stdout=PIPE).stdout.read()

def popenCall(xcmd=None,xin=None,bufsize=1024,shell=True,debugTF=False):
	""" Description: system call via subprocess.Popen """
	if not xcmd:
		return None,None
	if debugTF:
		sys.stderr.write("==SHELL_CMD:\n{}\n".format(xcmd))
		if xin:
			sys.stderr.write("  input:\n{}\n".format(xin))
	p = Popen(xcmd,shell=shell,bufsize=bufsize,stdin=PIPE,stdout=PIPE,stderr=PIPE)
	xin = (xin+"\n").encode('utf-8') if xin else None
	o, e = p.communicate(input=xin)
	return o.decode(), e.decode()

def str2float(s,r=None):
	'''
	convert string 's' to a valid float or 'r'=None
	'''
	try:
		return float(s)
	except Exception as e:
		return r

def strc2float(s,r=None,sc=','):
	'''
	convert comma string 's' to a valid float or 'r'=None
	
	'''
	if len(sc)>0:
		s = re.sub('[{}]'.format(sc),'',s)
	return str2float(s,r)

def str2value(s,r=None):
	'''
	convert string 's' to a valid numeric value or 'r'=None
	'''
	v = re.search("([+-]?\d*[.]?\d*)([eE][+-]?\d\d*)?",s).groups()
	if len(v[0])>0:
		return str2float(s,r)
	else:
		return r

def num_en2cn(s='10B',numTF=False):
	''' convert 's' from '10B' into '100億'
	'''
	v = re.search("[+-]?(\d*[.]?\d*)\s?(\w)",s)
	n,u = v.groups()
	u = u.upper()
	n = float(n)
	if u == 'B':
		x = "{:.0f}{}".format(n*10,'億')
		n = n*10**9
	elif u == 'M':
		x  = "{:.0f}{}".format(n*100,'萬')
		n = n*10**6
	elif u == 'T':
		x = "{}{}".format(n,'兆')
		n = n*10**12
	else:
		x = "{}".format(n)
	if numTF:
		return n
	return x
	

def sub_special_char(s='',p='{}[]()#%|*,:;/ ',n=''):
	"""
	covert string 's' that contains character in 'p' string and replace each one with 'n'
	"""
	import re
	rx = '[' + re.escape(''.join(p)) + ']'
	return re.sub(rx,n, s)

def consecutive_sign(v,zeroSign=0,reverseTF=True):
	"""
	Calc (number of same consecutive sign, sign) from the end
	Note,
	zeroSign can be treated as [-1,0,1]
	"""
	if len(v)<2:
		return 1,0
	if reverseTF is True:
		xv = v[::-1]
	else:
		xv = v
	if zeroSign==1:
		xv=np.where(np.array(xv)>=0,1,-1)
	elif zeroSign==-1:
		xv=np.where(np.array(xv)>0,1,-1)

	s = np.sign(xv).astype(int)
	b = s[0]
	for j,x in enumerate(s[1:]):
		if x!=b:
			j=j+1
			break
	return j, b

def run_jj2(ts,dd={},dirname='.',fileTF=False,debugTF=False):
	"""
	run jinja2 template string [ts] to output text [jj_str]
	where: 
		ts: template string
		**dd:  rendered inputs in {}
	Note: 
		[ts] is a template filename in [dirname] if fileTF is True
	"""
	#from jinja2 import Template as jjTemplate
	j2Env = jinEnv(loader=FileSystemLoader(dirname),trim_blocks=True,extensions=['jinja2.ext.do'])
	if debugTF:
		sys.stderr.write("=====run_jj2():\nfileTF:{},debugTF:{}\nts:{}\n".format(fileTF,debugTF,ts) )
		sys.stderr.write("dd:{}\n".format(dd) )
	if not ts:
		''
	if fileTF is False:
		return(j2Env.from_string(ts).render(**dd))
		#return jjTemplate(ts).render(**dd)
		#no loader environment specified for jjTemplate
	else:
		return(j2Env.get_template(ts).render(**dd))

def jj2_fmt(ts,dirname='.',fileTF=False,debugTF=False,j2onlyTF=False,**dd):
	return jj_fmt(ts,dd,dirname=dirname,fileTF=fileTF,debugTF=debugTF,j2onlyTF=j2onlyTF)

def get_arg2func(funcName,**optx):


	'''
	return a function-call or None based on 'funcArg'
	'''
	funcArg=None
	if isinstance(funcName,str):
		if funcName in globals() and hasattr(globals()[funcName],'__call__'):
			funcArg =  globals()[funcName]
	return funcArg

def get_funcLst(unLst=['x_','Call','OLD','_DEPRECATED','Popen']):
	henv={ k:v for (k,v) in globals().items() if callable(v) and not any(map(k.__contains__,unLst))}
	return henv

@saferun
def jj_fmt(ts,dd={},dirname='.',fileTF=False,debugTF=False,j2onlyTF=False,**opts):
	"""
	run jinja2 template string [ts] to return output text [jj_str]
	based on input **dd + buildins + _alan_str defined functions
	where: 
		ts: template string
		dd:  rendered inputs in {}
	Note: 
		[ts] is a template filename in [dirname] if fileTF is True
	"""
	try:
		if ts is None:
			return ''
		elif len(ts)<1:
			return ''
		inisd = int(datetime.datetime.today().strftime("%m%d")) + hash(ts)%10000
		random.seed(inisd)
		#- Add available embedded function
		#unLst = ['sysCall','popenCall','_DEPRECATED','x_']
		#tpLst = (types.BuiltinFunctionType,types.FunctionType)
		#genv = { k:v for (k,v) in globals().items() if isinstance(v,tpLst) is True }
		#henv = {ky:va for ky,va in genv.items() if not any(map(ky.__contains__,unLst))}
		henv = get_funcLst()

		# Inherit variable from 'dd':
		d = dd.copy()
		# Add built-in functions:
		d.update(__builtins__)
		# Add Internal functions inside '_alan_str':
		d.update(henv) # include all the functions
		# Add additional variables from **opts:
		if len(opts)>0:
			d.update(opts)
		# Update debugTF and dirname variables:
		d.update(debugTF=debugTF,dirname=dirname)
		ctrlSTRUCT=getKeyVal(opts,'ctrlSTRUCT','')
		if ctrlSTRUCT is not None and ctrlSTRUCT in ['include','extends']:
			ts='{{% {} "{}" %}}'.format(ctrlSTRUCT,ts)
		if debugTF:
			sys.stderr.write( "@: {}\n".format(datetime.datetime.now()) )
			sys.stderr.write( "jj2 keys: {}\n".format(sorted(d.keys())) )
			sys.stderr.write( "Interal Func:\n{}\n".format(sorted(henv.keys())) )
			sys.stderr.write( "jj2 ts:\n{}\n".format(ts) )
			sys.stderr.write( "jj2 d:\n{}\n".format(d) )
		# Need at least one of tags ['{{' , '{%' ] in ts to run jinja2
		if '{{' in ts or '{%' in ts or fileTF is True or j2onlyTF is True:
			jj_str =  run_jj2(ts,dd=d,dirname=dirname,fileTF=fileTF,debugTF=debugTF)
		else: 
			jj_str = ts.format(**d)

		if jj_str is None:
			jj_str=''
		elif hasattr(jj_str,'__len__') and len(jj_str)>0:
			jj_str = ''.join([s for s in jj_str.splitlines(True) if s.strip()])
	except Exception as e:
		jj_str = "**ERROR @ {}():{}\nTS:\n{}".format('jj_fmt',str(e),ts[:80])
	return(jj_str)

def ifelse(yn,a,b):
	return a if yn else b

def clip(a,x,b):
	return min(max(a,x),b)

def isclip(a,x,b):
	return (True if a<=x and x<=b else False)

def isclipL(a,x,b):
	return (True if a<=x and x<b else False)

def isclipR(a,x,b):
	return (True if a<x and x<=b else False)

def num2MP3(s,dotSign='點',lang="cn",mp3YN=True):
	""" convert number str to audio readable format
	"""
	#TBD, buggy codes
	#if lang=='cn':
	#	dotSign='點'
	#	s = num_en2cn(s)
	s=str(s).replace(',','').strip()
	if '.' not in s or mp3YN==False:
		return s
	else:
		xs=s.split('%')
		pctSign='%' if len(xs)>1 else ''
		x=xs[0].split('.')
		if int(x[1])==0:
			return "{}{}".format(x[0],pctSign)
		if lang=="cn": 
			dotSign='點'
			y="{0}{3} {1} {2}".format(x[0],dotSign," ".join(x[1]),pctSign)
		else:
			dotSign="point"
			y="{0} {1} {2}{3}".format(x[0],dotSign," ".join(x[1]),pctSign)
		return y

def gsrg(r,n):
	""" get signicant_rounding digit """
	return int(max(ceil(log(abs(r),.1)),n)) if abs(r)>0 else n

def roundTBM(r,n):
	""" 
	rounding r to nearest [T|B|M] and its (n, signicant_digit)
	"""
	if r>10**12:
		xstr="{:.{}f}T".format(r/10.0**9,n)
	elif r>10**9:
		xstr="{:.{}f}B".format(r/10.0**9,n)
	elif r>10**6:
		xstr="{:.{}f}M".format(r/10.0**6,n)
	else:
		xstr="{:.{}f}".format(r,n)
	return xstr

def roundSD(r,n,lang='cn',mp3YN=False,TBMTF=False): 
	""" rounding r to max (n, signicant_digit)
	"""
	n=gsrg(r,n)
	if TBMTF:
		xstr = roundTBM(r,n)
	else:
		xstr="{:.{}f}".format(r,n)
	if mp3YN==True:
		xstr=num2MP3(xstr,lang=lang,mp3YN=mp3YN)
	return xstr

def roundPctMP3(r,n=0,pct=100,dotSign='.',lang='en'):
	n=gsrg(r*pct,n)
	s="{:.{}f}%".format(r*pct,n)
	if n<1:
		return s
	else:
		x=s[:-1].split(".")
		if int(x[1])==0:
			return "{}{}".format(x[0],"%")
		if lang=="cn":
			dotSign='點'
			y="{} {} {}{}".format(x[0],"%",dotSign," ".join(x[1]))
		else:
			y="{} {} {}{}".format(x[0],dotSign," ".join(x[1]),"%")
		return y

def roundPct(r,n=0,pct=100,dotSign='.',lang='en',mp3YN=False):
	"""
	return (r*100)% rounding to nearest n decimal
	Note,
		dotSign prints decimal point '.' for mp3 sound
		if mp3 flag mp3YN is True
	"""
	if mp3YN is True:
		return roundPctMP3(r,n=n,pct=pct,dotSign=dotSign,lang=lang)
	n=gsrg(r*pct,n)
	return "{:.{}f}%".format(r*pct,n)

def loc_dindex(r,thd=[1.96,1.15,0.32,0.1,-0.1,-0.32,-1.15,-1.96],openTF=True,sortTF=False):
	"""
	Locate the 1st index of descending thresholds:thd for r > thd[index] 
	Return:
		index location
	Note:
		return the 1st index of r >= thx[index] if openTF = False
	"""
	n = len(thd)
	if sortTF is True:
		thd = np.sort(thd)[::-1]
	if openTF is False:
		#ul = filter(lambda x: thd[x]-r <=0, range(n))[:1]
		ul = [ x for x in range(n) if thd[x]-r<=0 ][:1]
	else:
		#ul = filter(lambda x: thd[x]-r < 0, range(n))[:1]
		ul = [ x for x in range(n) if thd[x]-r<0 ][:1]
	return ul[0] if len(ul)>0 else n

def loc_aindex(r,thd=[-1.96,-1.15,-0.32,-0.1,0.1,0.32,1.15,1.96],openTF=True,sortTF=False):
	"""
	Locate the 1st index of ascending thresholds:thd for r > thd[index] 
	Return:
		index location
	Note:
		return the 1st index of r >= thx[index] if openTF = False
	"""
	n = len(thd)
	if sortTF is True:
		thd = np.sort(thd)
	if openTF is True:
		ul = list(filter(lambda x: thd[x]-r >=0, range(n)))[:1]
	else:
		ul = list(filter(lambda x: thd[x]-r > 0, range(n)))[:1]
	return ul[0] if len(ul)>0 else n

def udfSentence(tkName='AAPL',tkVal=999,tkChg=0,tkChgPct=0,wordLst=None,thd=None,lang='cn',openTF=True,sortTF=False,ascendingTF=False,scale=1,pct=1,mp3YN=False):
	"""
	Return directional descripting sentence based on threshold: thd 
	Inputs: tkName,tkVal,tkChg,tkChgPct(in %)
	Example: udfSentence(tkName='道瓊',tkVal=35000,tkChg=2,tkChgPct=-1.2)
	"""
	dx=locals().copy()
	tmplstr_cn="""
{{tkName}}{{udfWord(tkChgPct,lang=lang)}}
{%- if abs(tkChgPct)>0.5 -%}
	{{roundPct(tkChgPct,0,pct=pct,lang=lang,mp3YN=mp3YN)}}
{%- elif abs(tkChg)>=0.5 -%}
	{{roundSD(tkChg,0,lang=lang,mp3YN=mp3YN)}}點
{%- endif -%}
，{{rdWord('來到|達|為')}}{{roundSD(tkVal,0,lang=lang,mp3YN=mp3YN)}}。
"""
	tmplstr_en="""
{{tkName}} {{udfWord(tkChgPct,lang=lang)}} {{''}}
{%- if abs(tkChgPct)>0.5 -%}
	{{roundPct(tkChgPct,0,pct=pct,lang=lang,mp3YN=mp3YN)}}
{%- elif abs(tkChg)>=0.5 and abs(tkChgPct)>0.032 -%}
	{{roundSD(tkChg,0,lang=lang,mp3YN=mp3YN)}}
{%- endif -%}
, {{rdWord('reached to|at')}} {{roundSD(tkVal,0,lang=lang,mp3YN=mp3YN)}}. 
"""
	tmplstr=tmplstr_cn if lang=='cn' else tmplstr_en
	return jj_fmt(tmplstr,dx)

def udfWord(r,wordLst=None,thd=None,lang='cn',openTF=True,sortTF=False,ascendingTF=False,scale=1):
	""" 
	Directional descriptor of r based on threshold: thd 
	Where
	  r: float or int value
	  thd[]: threadhold numeric array
	  wordLst[]: a list of action word w.r.t thd[index]
	Return:
	  Directional verb/adjective
	Note:
	  wordLst[] array should be 1+ length of thd[]
	  thd: threshold buckets
	  default to [1.96,1.15,0.16,0.032,-0.032,-0.16,-1.15,-1.96] multiplied by 'scale'
	Example: udfWord(1.4,lang='cn')
	"""
	if thd is None:
		thd = [1.96,1.15,0.16,0.032,-0.032,-0.16,-1.15,-1.96]
		if scale != 1:
			thd = [x*scale for x in thd ]
	if wordLst is None and lang=='cn':
		xstr="""
強勢上揚大漲|大幅升高走強|強勢攀升上漲|大幅上揚升高
明顯走高|明確向上攀升|強勢上揚
上升|攀升|上漲
小幅上漲|微漲|微微攀升|小幅微升
上下微動|輕微震盪|區間微震，幾無變化|稍稍波動，幾無變化|沒什麼變動
微微下降|微降|微微走低|小幅走低|稍稍下跌|小幅走跌
下跌|下降|走低|走弱|走跌
明顯向下走低|清楚顯現跌勢|明白顯露疲弱態勢
重跌下挫|跳樓暴跌|重挫走跌|跳水大跌|深跌入谷|大幅走跌下挫
		"""
		wordLst = xstr.strip().splitlines()
	elif wordLst is None:
		wordLst = ['advanced', 'gained', 'increased', 'rose', 'was quite flat', 'slipped', 'decreased', 'lose', 'dropped']
	if ascendingTF is True:
		idx = loc_aindex(r,thd=thd,openTF=openTF,sortTF=sortTF)
	else:
		idx = loc_dindex(r,thd=thd,openTF=openTF,sortTF=sortTF)
	if idx < len(wordLst):
		xlst = (unicode, str) if hasattr(__builtins__,"unicode") else str
		if isinstance(wordLst[idx],xlst):
			xstr =  rdWord(wordLst[idx])
		else:
			xstr = wordLst[idx]
	else:
		xstr = ''
	return rdWord(xstr)

def performance_verb(r,udfLst):
	n=len(udfLst.thd)
	ul=list(filter(lambda x:udfLst.thd[x]-abs(r)<=0, range(n)))[:1]
	ux = n-1 if len(ul)<1 else ul[0]
	udf_str=udfLst.loc[ux,["u","d","f"]]
	xstr= udfStr(r,udf_str)
	return xstr

def btwDates(a,b,bracket=None,fmt=None,lang="en"):
	if lang.lower() == "cn" :
		return btwDates_cn(a,b,bracket,fmt)
	else :
		return btwDates_en(a,b,bracket,fmt)

def btwDates_cn(a,b,bracket=None,fmt=None):
	if bracket is None :
		bracket=["在于","和","之間"]
	if fmt is None :
		fmt="%Y年%m月"
	s=""
	a=ymd2ym(a,ym=fmt)
	b=ymd2ym(b,ym=fmt)
	return s.join([bracket[0],a,bracket[1],b,bracket[2]])

def btwDates_en(a,b,bracket=None,fmt=None):
	if bracket is None :
		bracket=["between","and"]
	if fmt is None :
		fmt="%Y-%m"
	s=" "
	a=ymd2ym(a,ym=fmt)
	b=ymd2ym(b,ym=fmt)
	return s.join([bracket[0],a,bracket[1],b])

def datetime2hm(dtm,dformat="%H:%M",lang="cn",mp3YN=False):
	if lang!="cn":
		return dtm.strftime(dformat)
	dformat = dformat.replace("%d","%d日").replace("-","")
	if mp3YN == True:
		dformat = dformat.replace(":","")
		dformat = dformat.replace('%H',"%H點")
		dformat = dformat.replace('%M',"%M分") if dtm.minute>0 else  dformat.replace('%M','整') 
		dformat = dformat.replace('%S',"%S秒") if dtm.second>0 else  dformat.replace('%S','')
	return dtm.strftime(dformat)

def ymd2datetime(s,dformat="%Y%m%d"):
	if (isinstance(s,(np.integer,int,float)) or s.isdigit() ):
		s = int(s)
		if log(s,10) >= 12:
			s = s/1000
		if log(s,10) >= 9:
			return datetime.datetime.fromtimestamp(s)
	return datetime.datetime.strptime(str(s),dformat)

def ymd2ym(s,ym=None,ymd='%Y-%m-%d',lang="cn"):
	if ym is None:
		ym = "%Y年%m月" if lang=="cn"  else "%Y-%m"
	return ymd2datetime(s,ymd).strftime(ym)

def ymd2ymd(s,ym=None,ymd='%Y-%m-%d',lang="cn",):
	if ym is None:
		ym = "%Y年%m月%d日" if lang=="cn"  else "%Y-%m-%d"
	if isinstance(s, datetime.date):
		return s.strftime(ym)
	else:
		return ymd2datetime(s,ymd).strftime(ym)

def ymd2md(s,ym='%m-%d',ymd='%Y%m%d',lang="en",mp3YN=False):
	if isinstance(s, datetime.date):
		dtm = s
	else:
		dtm = ymd2datetime(s,ymd)
	if lang=="cn":
		ym = ym.replace("%d","%d日").replace("-","")
		vdu = list(filter(lambda x: x in ym, 'mbB'))
		if len(vdu)>0:
			xdu = '%'+vdu[0]
			ym = ym.replace(xdu,"%m月")
		if mp3YN == True:
			ym = ym.replace(":","")
			ym = ym.replace('%H',"%H點")
			ym = ym.replace('%M',"%M分") if dtm.minute>0 else  ym.replace('%M','整') 
			ym = ym.replace('%S',"%S秒") if dtm.second>0 else  ym.replace('%S','')
	else:
		locale.setlocale(locale.LC_TIME,"en_US.UTF8")
		if ym is None or len(ym)<4:
			ym='%m-%d'
	return dtm.strftime(ym)

def pattern_cn(name,lang="cn"):
	ptnObj={"morning_star":"晨星","evening_star":"夜星",
	"bullish_harami":"曙光乍現","bearish_harami":"烏雲蓋頂",
	"3_white_soldiers":"紅三兵","3_black_crows":"黑三兵",
	"MACD":"MACD看漲" , "bearish_MACD":"MACD看跌" }
	if lang.lower() == "cn" and name in ptnObj:
		return ptnObj[name]
	else :
		return name

def fq2unit_str(fqStr,lang="cn"):
	""" return unitStr from fqStr
	"""
	if lang.lower() == "cn" :
		unitObj={"D":"個交易日","W":"週","M":"個月","Q":"季","Y":"年"}
	else :
		unitObj={"D":"day","W":"week","M":"month","Q":"quarter","Y":"year"}
	return unitObj[fqStr] if fqStr.upper() in unitObj else unitObj["D"]

def rdWord(wordLst,sep="|"):
	""" Select random word from a word list separated with "|" pipe delimeter 
		Args:
			wordLst: list of words glue with "|"
			sep: word separator 
		Returns:
			random selected word
	"""
	xu=wordLst.split(sep)
	return(random.choice(xu))

def udfSet(udf=None,lang="en",tense="past"):
	if lang == "en":
		udx=["increased","decreased","been flat"] if tense=="past" else ["increases","decreases","is basically flat"]
	else:
		udx=["上升|攀升|上漲","下跌|下降","上下稍稍波動|上下輕微震盪|區間微微震盪"]
	if udf is None:
		return udx
	elif type(udf) == str:
		udf = [udf]
	ns=min(len(udf),len(udx))
	for j in range(ns):
		udx[j]=udf[j]
	return udx 

#def udf_word(r,udf=None,zs=[0,0],lang="en",tense="past"):
#	return udfStr(r,udf=udf,zs=zs,lang=lang,tense=tense)

# ['\u2191','\u2193','']

def udfSym(r,udf=[u'\u25B2',u'\u25BC',''],zs=0):
	if  (r>zs):
		xu=udf[0]
	elif(r<zs):
		xu=udf[1]
	else:
		xu=udf[2]
	return xu

def udfStr(r,udf=None,zs=[0,0],lang="en",tense="past"):
	""" 
	Directional descriptor based on value: r 
		r>zs[0], within [ zs[0],zs[1] ], zs[1]>r
		Args:
			r: float or int value
			udf: a list of 3 words 
			zs: threshold values 
		Returns:
			Directional verb/adjective
	"""
	try:
		udx=udfSet(udf,lang=lang,tense=tense)
		if isinstance(zs,(np.integer,int,float)) is True:
			z = abs(zs)
			azs=[z,-z]
		elif len(zs)<1: 
			azs=[0,0]
		elif len(zs)<2: 
			z = abs(zs)
			azs=[z,-z]
		else:
			azs=zs
		if  (r>azs[0]) :
			xu=udx[0]
		elif(r<azs[1]) :
			xu=udx[1]
		else :
			xu=udx[2]
		return(rdWord(xu))
	except Exception as e:
		return(str(e))
	
def udfStr_DEPRECATED(r,udf=None,zs=0,lang="en",tense="past"):
	""" DEPRECATED
	Directional descriptor based on value:r  [r>zs,r<-zs,in-between]
		Args:
			r: float or int value
			udf: a list of 3 words 
			zs: threadhold value
		Returns:
			Directional verb/adjective
	"""
	try:
		udx=udfSet(udf,lang=lang,tense=tense)
		azs=abs(zs)
		if  (r>azs) :
			xu=udx[0]
		elif(r<-azs) :
			xu=udx[1]
		else :
			xu=udx[2]
		return(rdWord(xu))
	except Exception as e:
		return(str(e))
	
def roundMP3(x,n=0,usdSign=None,dotSign='.',lang=None):
	""" Audio format expression
		Args:
			x: Float or int value
			n: Rounding decimal
			usdSign: not used, only for being compatible with ruondUSD
			dotSign: 'point' for reading purpose
	"""
	if n<=0:
		return '{:.0f}'.format(x)
	else:
		ri,rj = '{:.{}f}'.format(x,n).split(".")
		rmd = dotSign + ' '.join(rj) if int(rj)>0 else ''
		return '{}{}'.format(ri,rmd)

def roundAuto(x,n=2,usdSign=None,dotSign='.',lang='en',mp3YN=False):
	if isinstance(x,(list,np.ndarray)):
		return [roundUSD(d,n=n,usdSign=usdSign,dotSign=dotSign,lang=lang,mp3YN=mp3YN) for d in x]
	return roundUSD(x,n=n,usdSign=usdSign,dotSign=dotSign,lang=lang,mp3YN=mp3YN)
	
def roundUSD(x,n=0,usdSign='$',dotSign='.',lang='en',mp3YN=False):
	""" Accounting format expression
		Args:
			x: Float or int value
			n: Rounding decimal
			usdSign: Dollar sign
		Returns:
			Accounting-format string
		Note:
			show simple rounded number if usdSign is None
	"""
	if mp3YN is True:
		xstr = roundSD(x,n=n,lang=lang,mp3YN=mp3YN,TBMTF=True)
		if lang=='cn':
			usdSign='美元'
			xstr = "{}{}".format(xstr,usdSign)
		else:
			xstr = "{}{}".format(xstr,usdSign)
		#return roundMP3(x,n=n,usdSign=usdSign,dotSign=dotSign,lang=lang)
		return xstr
	elif float(x)>10**6:
		xstr = roundSD(float(x),n=n,lang=lang,mp3YN=mp3YN,TBMTF=True)
		xf = usdSign + num_en2cn(xstr) if lang=='cn' else xstr
		return xf
	if n is None: # single digit or less round to 1 decimal else round to integer
		n=0 if abs(x)>=10 or abs(x)%1<0.05 else 1
	else: # if rounding digits are 0s do not show
		n=0 if abs(x)%1<0.5/10**int(n) else n
	if usdSign is None:
		if n<0: # use significant precision if precision n is negative
			xf='{:.g}'.format(x)
		else:
			xf='{:.{}f}'.format(x,n)
	else:
		xf='{}{:0,.{}f}'.format(usdSign,x,n)
	return(xf)

def wrapWord(a,bk=["(",")"]):
	return '{1}{0}{2}'.format(*([a]+bk))

def rangeStr(a,b,bk=["[","]"]):
	return '{2}{0}, {1}{3}'.format(*([a,b]+bk))

def btwStr(a,b,bk=["ranging between","and"],fl="fluctuated around"):
	if a==b:
		return "{} {}".format(fl,a)
	return '{2} {0} {3} {1}'.format(*([a,b]+bk))

def btwStr_cn(a,b,bk=["介于","至","的範圍之間"],fl=["變化不大，只會在","內上下稍稍波動"],unitStr=''):
	if a==b:
		return "{2} {0}{1}{3}".format(*([a,unitStr]+fl))
	return '{2} {0} {3} {1} {5}{4}'.format(*([a,b]+bk+[unitStr]))

def connMdb(clientM=None):
	if isinstance(clientM,MongoClient):
		return clientM
	elif clientM is None: # as None
		uriM = "localhost:27017"
		clientM=MongoClient(uriM)
	elif isinstance(clientM,str): # as hostname
		uriM = "{}:27017".format(clientM)
		clientM=MongoClient(uriM) # existed connection 
	else:
		return clientM
	return clientM

def find_mdb(jobj={},clientM=None,dbname='ara',tablename=None,field={},sortLst=[],ascendingTF=False,limit=0,dfTF=False,**optx):
	""" find WHERE: jobj from mongodb [dbname]::[tablename]
	 and display FIELD: field
		Args:
			a json object
		returns:
			jobj if success
	"""
	from bson import json_util
	mobj=[]
	try:
		clientM=connMdb(clientM)
		if tablename is None:
			return jobj,clientM,"tablename not defined!"
		if hasattr(clientM,'_Database__client') and isinstance(clientM._Database__client,MongoClient):
			dbM=clientM
		else:
			dbM=clientM[dbname]
		zobj={k:1 for k in field } if len(field)>0 else {}
		if len(sortLst)>0 and isinstance(sortLst,dict):
			# Note, sorting might be diff as dict variables are not in order
			zsrt=[(k,v) for k,v in sortLst.items()]
		elif len(sortLst)>0 and isinstance(sortLst[0],(tuple,list)):
			zsrt=sortLst
		elif ascendingTF:
			zsrt=[(k,1) for k in sortLst] if len(sortLst)>0 else [('_id',1)]
		else:
			zsrt=[(k,-1) for k in sortLst] if len(sortLst)>0 else [('_id',-1)]
		if '_id' not in field:
			zobj.update(_id=0)
		zobj=json.loads(json.dumps(zobj))
		if limit>0:
			mbson = dbM[tablename].find(jobj,zobj,sort=zsrt).limit(limit)
		else:
			mbson = dbM[tablename].find(jobj,zobj,sort=zsrt)
		#mobj= json.loads(json_util.dumps(mbson))
		mobj= [x for x in mbson]
		clientM.close()
		msg="success"
	except Exception as e:
		msg=str(e)
	if len(mobj)>0 and len(mobj[0])<1:
		mobj=[]
	if dfTF:
		mobj = pd.DataFrame(mobj)
	return mobj,clientM,msg

def get2mndb(jobj,tablename='lsi2nlg',zpk={'ticker','username','tmplname','category','tmpltype','lang'}):
	""" find jobj from mongodb [ara:ls2nlg] based on 'username' & 'tmplname' keys 
		Args:
			jobj:json object
			zpk: set of primary keys 
		returns:
			jobj with 'retcode' updated
	"""
	try:
		uriM = "localhost:27017"
		clientM=MongoClient(uriM)
		dbM=clientM['ara']
		
		zobj={k: jobj[k] for k in set(jobj.keys()) & zpk } if len(zpk)>0 else jobj
		# find via 'ticker','username','tmplname','category','tmpltype','lang'
		mobj=dbM[tablename].find_one(zobj)
		# find via 'username','tmplname','category','tmpltype','lang'
		if mobj is None:
			if 'ticker' in zobj:
				del zobj['ticker']
			mobj=dbM[tablename].find_one(zobj)
		# find via 'username','category','tmpltype','lang'
		if mobj is None:
			zobj['username'] = 'system'
			if 'tmplname' in zobj:
				del zobj['tmplname']
			mobj=dbM[tablename].find_one(zobj)
		if mobj is None:
			jobj['retcode']='No record found on {}'.format(zobj.items())
			return jobj
		else: 
			mobj['username'] = jobj['username'] if 'username' in jobj else 'system'
		if len(jobj)>1:
			for j,xky in enumerate(jobj):
				if xky not in mobj:
					continue
				jobj[xky]= mobj[xky]
		else:
			jobj.update(mobj)
		clientM.close()
		if 'tmplrpt' not in jobj:
			jobj['tmplrpt']= ''
		if 'lastmod' in mobj:
			jobj['retcode'] += " Last mod. @ {}".format(mobj['lastmod'])
		else:
			jobj['retcode']='Record found via {}'.format(zobj.items())
		sys.stderr.write("====={}\n".format(jobj))
	except Exception as e:
		jobj['retcode']='**ERROR: {} @ get2mndb()'.format(str(e))
	return jobj

def upsert_mdb(jobj={},clientM=None,port=27017,dbname='ara',tablename=None,wmode='upsert',zpk=[],zasc=[],**optx):
	"""
	upsert 'jobj' to MDB 'dbname'::'tablename'
	  w.r.t unique primary keys 'zpk' defined as zip('zpk','zasc')
	Optional parameters: 
	  pkTF=True  : To create primary keys
	  uniqueTF=True : To ensure unique primary key
	  ordered=False : to ensure insert even duplicate errors occur 
	Example:
	upsert_mdb({"name":"Mary"},tablename="test",zpk=["name"])
	upsert_mdb({"name":"Mary"},tablename="test",zpk=["name"],pkTF=False)
	"""
	return insert_mdb(jobj=jobj,clientM=clientM,port=port,dbname=dbname,tablename=tablename,wmode=wmode,zpk=zpk,zasc=zasc,**optx)

# upsert_mdb wrapper
def write_mdb(jobj={},clientM=None,port=27017,dbname='ara',tablename=None,wmode='upsert',zpk=[],zasc=[],**optx):
	return insert_mdb(jobj=jobj,clientM=clientM,port=port,dbname=dbname,tablename=tablename,wmode=wmode,zpk=zpk,zasc=zasc,**optx)

def insert_mdb(jobj={},clientM=None,port=27017,dbname='ara',tablename=None,wmode='',zpk=[],zasc=[],**optx):
	"""
	Insert 'jobj' to MDB 'dbname'::'tablename'
	  w.r.t unique primary keys 'zpk' defined as zip('zpk','zasc')
	Optional parameters: 
	  pkTF=True  : To create primary keys
	  uniqueTF=True : To ensure unique primary key
	  ordered=False : to ensure insert even duplicate errors occur 
	Example:
	insert_mdb({"name":"Mary"},tablename="test",zpk=["name"])
	insert_mdb({"name":"Mary"},tablename="test",zpk=["name"],pkTF=False)
	"""
	if not all([len(jobj),tablename]):
		errmsg="**ERROR:{} @{}\n".format(tablename,"insert_mdb()")
		return jobj,None,errmsg
	try:
		pkTF=getKeyVal(optx,'pkTF',True)
		uniqueTF=getKeyVal(optx,'uniqueTF',True)
		ordered=getKeyVal(optx,'ordered',False)
		debugTF=getKeyVal(optx,'debugTF',False)
		createTF=getKeyVal(optx,'createTF',False)
		clientM=connMdb(clientM)
		if hasattr(clientM,'_Database__client') and isinstance(clientM._Database__client,MongoClient):
			mdb=clientM
		else:
			mdb=clientM[dbname]
		try:
			if createTF is True or wmode=='create':
				ynX=mdb[tablename].drop()
				sys.stderr.write("Drop collection:{}, status:{}\n".format(tablename,ynX))
			if len(zpk)>0 and pkTF is True:
				if any( [ len(zasc)<1, len(zasc)!=len(zpk) ]):
					zasc=[1]*len(zpk)
				pkLst = [(k,v) for k,v in zip(zpk,zasc)]
				mdb[tablename].drop_indexes()
				mdb[tablename].create_index(pkLst,unique=uniqueTF)
				emsg="==Indexing primary keys pkLst:{} @{}\n".format(pkLst,"insert_mdb")
				sys.stderr.write(emsg)
		except Exception as e:
			errmsg="**ERROR:{} @{}\n".format(str(e),"insert_mdb:create_index")
			sys.stderr.write(errmsg)
			pass
		if isinstance(jobj,pd.DataFrame):
			jobj = jobj.to_dict(orient='records')
		elif isinstance(jobj,dict):
			jobj = [jobj]
		if getKeyVal(optx,'jsonObjReloadTF',False):
			jobj = jsonObjReload(jobj)
			sys.stderr.write(" --{} @ {}\n".format('jsonObjReload','insert_mdb()'))
		if wmode=='upsert':
			if debugTF:
				sys.stderr.write(" --upsert:{}\n".format("upd2mdb()"))
			ret=upd2mdb(jobj,zpk=zpk,mcur=mdb[tablename],debugTF=debugTF,)
		else:
			if wmode=='replace':
				mdb[tablename].delete_many({})
			ret=mdb[tablename].insert_many(jobj,ordered=ordered)
		errmsg="{}".format(ret)
	except Exception as e:
		errmsg="**ERROR:{} @{}\n".format(str(e),"insert_mdb:insert_many")
		sys.stderr.write(errmsg)
		try:
			sys.stderr.write("---JOBJ[0]:\n{}\n".format(jobj[0]))
		except:
			pass
		return {},None,errmsg
	return jobj,mdb,errmsg

def save2mgdb(mLst=None,mky=None,dbM=None,tablename="temp",wmode='replace'):
	if dbM is None or mLst is None:
		return None
	createTF=True if wmode=='replace' else False
	dy,_,errmsg=insert_mdb(jobj=mLst,zpk=mky,dbM=dbM,tablename=tablename,createTF=createTF)
	return len(dy)

def upd2mdb(dd=[],zpk={},mcur={},debugTF=False):
	''' support function to insert_mdb
	mongodb utility of insert_mdb/upsert_mdb
	remove and insert for update based on zpk primary key
	'''
	def remove_doc(jobj,zpk,nz):
		ret={}
		if nz<1:
			return ret
		jx = { ky:va for ky,va in jobj.items() if ky in zpk }
		if len(jx)==nz:
			if debugTF:
				sys.stderr.write(" --remove primany key doc:{}\n".format(jx))
			ret = mcur.remove(jx)
		elif len(jx)<nz:
			sys.stderr.write(" --{} missing primary keys:{}\n".format(jx,zpk))
		return ret

	nz = len(zpk)
	for j, jobj in enumerate(dd):
		try:
			if debugTF:
				sys.stderr.write(" --insert doc:{}:{}:{}\n{}\n".format(type(jobj),jobj.keys(),mcur,jobj))
			jobj.pop('_id',None)
			remove_doc(jobj,zpk,nz)
			ret = mcur.insert(jobj)
		except Exception as e:
			errmsg="**ERROR:{}:{} @{}\n{}\n".format(j,str(e),"upd2mdb()",jobj)
			sys.stderr.write(errmsg)
	return dd

def np2mongo(zobj):
	for j,(x,y) in enumerate(zobj.items()):
		if isinstance(y,np.integer):
			zobj[x] = int(y)
		elif isinstance(y,np.float):
			zobj[x] = float(y)
	return zobj

def write2mdb(jobj,clientM=None,dbname='ara',tablename=None,zpk={},insertOnly=False):
	"""
	write & replace jobj to mongodb [dbname]::[tablename]
	based on primary keys in {} dict format: {zpk}
	Args:
		jobj: a dataFrame, list_of_object or object
	returns:
		list_of_object, mongo+database_handler, error_msg
	Note: 
	  1. zpk={'*'}:  delete to any existed records before insertion 
	  2. insertOnly=True: insert record without checking any duplication
	  3. 'jobj' return with additional '_id' field after execution
	"""
	def jobj2mdbc(jobj,mdbc,zpk,insertOnly=False):
		mobj = jobj # jobj.copy()
		if mobj.get('_id',False):
			del mobj['_id']
		if insertOnly:
			mdbc.insert(mobj)
			return mobj
		try:
			if len(zpk)<1:
				xpk = {}
			else:
				xpk = set(mobj.keys()) & set(zpk)
			if len(xpk)>0 and len(xpk)>=len(set(zpk)):
				zobj={k: mobj[k] for k in xpk }
				try:
					zobj = np2mongo(zobj)
					mdbc.delete_many(zobj)
				except Exception as e:
					sys.stderr.write("**ERROR: {} @ {}, mdbc:{}, zpk:{}, zobj:\n{}\n".format(str(e),'delete_many()',mdbc,zpk,zobj) )
					sys.stderr.write("===rpt_hm: type:{},val:{}".format(type(zobj['rpt_hm']),zobj['rpt_hm']))
		except Exception as e:
			sys.stderr.write("**ERROR: {} @ {}, zpk:{}, jobj:\n{}\n".format(str(e),'jobj2mdbc()',zpk,jobj) )
		mobj = np2mongo(mobj)
		mdbc.insert(mobj)
		return mobj

	if clientM is None:
		uriM = "localhost:27017"
		dbM=MongoClient(uriM)[dbname]
	elif isinstance(clientM,str) is True:
		uriM = "{}:27017".format(clientM)
		dbM=MongoClient(uriM)[dbname]
	elif isinstance(clientM,MongoClient):
		dbM=clientM[dbname]
	elif hasattr(clientM,'_Database__client') and isinstance(clientM._Database__client,MongoClient):
		dbM = clientM
	elif isinstance(clientM,MongoClient) is False:
		return jobj,None,"no mongodb driver"
	mobj={}
	try:
		msg="success"
		if isinstance(jobj,list) is True:
			for xobj in jobj:
				try:
					mobj = jobj2mdbc(xobj,dbM[tablename],zpk=zpk,insertOnly=insertOnly)
				except Exception as e:
					sys.stderr.write("**ERROR: {} @ {}\n".format(str(e),'write2mdb() List-Dict') )
					msg=str(e)
					continue
		elif isinstance(jobj,pd.DataFrame) is True:
			df = jobj
			jobj = df.to_dict(orient='records')
			if len(zpk)>0 and set(zpk) == {'*'}:
				dbM[tablename].delete_many({})
				zpk = {}
			for xobj in jobj:
				try:
					mobj = jobj2mdbc(xobj,dbM[tablename],zpk=zpk,insertOnly=insertOnly)
				except Exception as e:
					sys.stderr.write("**ERROR: {} @ {}\n".format(str(e),'write2mdb() DataFrame') )
					msg=str(e)
					continue
		else:
			try:
				mobj = jobj2mdbc(jobj,dbM[tablename],zpk=zpk,insertOnly=insertOnly)
			except Exception as e:
				sys.stderr.write("**ERROR: {} @ {}, zpk:{}, jobj:\n{}\n".format(str(e),'write2mdb() Dict',zpk,jobj) )
				msg=str(e)
		#clientM.close()
	except Exception as e:
		#import pickle
		msg=str(e)
		sys.stderr.write("**ERROR: {} @ {}\n".format(msg,'write2mdb()') )
		#fp = open("mdb_err.pickle","wb")
		#pickle.dump(jobj,fp)
		#fp.close()
	return mobj,dbM,msg

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
	try:
		if isinstance(fp,str) is True:
			fp = sys.stdout if fp =='-' else open(fp,'w') 
		fp.write(s)
		ret = 1
	except Exception as e:
		sys.stderr.write("**ERROR: {} @ {}\n".format(str(e),'str_tofile()') )
		ret = 0
	return ret

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
	if df is None or len(df)<1:
		return ''
	if output == 'csv':
		ret = df.to_csv(sep=sep,index=indexTF,header=hdrTF)
	elif output == 'json':
		ret = df.to_json(orient='records')
	elif output == 'html':
		ret = df.to_html(index=indexTF,header=hdrTF)
	elif output == 'string':
		ret = df.to_string(index=indexTF,header=hdrTF)
	return ret

def save2mndb(jobj,zpk={'ticker','username','tmplname','category','tmpltype','lang'},clientM=None,dbname='ara'):
	""" 
	save jobj into mongodb [ara:ls2nlg] based on 'username' & 'tmplname' keys 
	Args:
		jobj: json object
		zpk: list of primary keys if exist
		clientM: mongo driver
		dbname: database name default to 'ara'
	returns:
		jobj with 'retcode' & 'lastmod' updated
	Note, create new table if zpk is {'*'} 
	"""
	try:
		mobj=jobj.copy()
		if jobj.get('retcode',False):
			del mobj["retcode"]
		del mobj["action"]
		if clientM is None:
			uriM = "localhost:27017"
			clientM=MongoClient(uriM)
		dbM=clientM[dbname]
		if set(zpk) == {'*'}:
			zobj = {}
			xpk = {}
		else:
			xpk = set(mobj.keys()) & set(zpk)
		if len(xpk)>0:
			zobj={k: mobj[k] for k in xpk }
		dbM.lsi2nlg.delete_many(zobj)
		dbM.lsi2nlg.insert(mobj)
		clientM.close()
		jobj["retcode"]="save successfully"
	except Exception as e:
		jobj["retcode"]="**ERROR:{} @ save2mndb()".format(e)
	return jobj

def lst2dict(jkst,jlst):
	""" combine 2 lists of shorter one to be a (key,value) object 
		Args:
			jkst: key list
			jlst: value list
		Returns:
			a (key,value) object
	"""
	jobj={}
	for j in range( min(len(jkst),len(jlst))):
		jobj[jkst[j]]=jlst[j]
	return jobj

def tmpl2lsi(tmplstr,argobj):
	""" Literal String Interpolation to format a text string
		Args:
			tmplstr: template text string
			argobj: a dictionary object 
		Returns:
			formatted interpolated text string
	"""
	if not argobj:
		return tmplstr
	else:
		return jj_fmt(tmplstr,argobj)

def combine_cmd2dict(*_xargs_): # TBD ,**_xdu_):
	""" combine list of scripts [_xargs_] and exec them  
	    return result as dict
	    Note: _xargs_, _xstr_ are reserved variables
	    Also see: qs_exec()
	"""
	try:
		_xstr_="\n".join(_xargs_)
		vux = qs_exec(_xstr_)
	except Exception as e:
		sys.stderr.write("**ERROR: {} @{}\n".format(str(e),"combine_cmd2dict"))
		return {}
	dux={}
	for x,v in vux.items():
		if isinstance(v, types.FunctionType) is True:
			continue
		elif x in ['_xargs_','_xstr_','_xdu_']:
			continue
		else:
			dux[x] = v
	sys.stderr.write("==Variables @{}():{}\n".format("combine_cmd2dict",dux))
	return dux

def jobj2lsi(jobj):
	""" Literal String Interpolation to format a text string
		Args:
			jobj: a dictionary object that contains 8 elements with keys:
			["tmplstr","argstr","prepstr","tmplrpt","username","category",
			"tmplname","action"]
			for interpolation argument list and pre-process
		Returns:
			formatted interpolated text string
	"""
	tmplrpt=""
	try:
		# combine script prepstr,argstr and exec them  
		if "prepstr" not in jobj:
			jobj["prepstr"]=''
		jobj["argobj"]=combine_cmd2dict(jobj["prepstr"],jobj["argstr"])
	except Exception as e:
		retcode="**ERROR: {} @{}() ".format(str(e),"combine_cmd2dict")
		return (tmplrpt,retcode)

	#- Run lsi to generate context
	try:
		tmplrpt=tmpl2lsi(jobj["tmplstr"],jobj["argobj"])
		retcode="run successfully"
	except Exception as e:
		retcode="**ERROR: {} @ tmpl2lsi() ".format(e)
	return (tmplrpt,retcode)

def lsi2nlg_calc(jobj):
	jobj["lastmod"]=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	#if jobj.keys()<8 :
	#	return jobj
	#elif jobj["action"] == 'save':
	if jobj["action"] == 'save':
		jobj=save2mndb(jobj)
	elif jobj["action"] == 'get':
		jobj=get2mndb(jobj)
	elif jobj["action"] == 'run':
		(jobj["tmplrpt"],jobj["retcode"])=jobj2lsi(jobj)
	return jobj

def _alan_nlg(debugTF=False):
	jobj={'username':'dan', 'prepstr':'global sq,sqt,quad\ndef sq(x):\n    return pow(x,2)\ndef sqt(x):\n    return pow(x,.5)\ndef quad(x):\n    return pow(x,4)', 'tmplrpt':'', 'tmplname':'global_macro_test1', 'tmplstr':'{label} {ticker} ({prc_cur}) has {VB_udf} from {prc_x1w} last week, we expect it will be up 3.19% next week, ranging between 1.21 and 1.41. Overall, 3-M Treasury (1.27) has increased 20.95% in the last 9 weeks. The latest significant move was an increase of 316.72% between May 2016 and August 2017.\n{y2},{y4},{y_5}', 'action':'run', 'category':'InterestRate', 'argstr':'label="3-M Treasury";ticker="DGS3MO";prc_cur=1.2805;prc_x1w=1.27;y2=sq(3); y4=quad(3); y_5=sqt(3);VB_udf=udfStr(prc_cur-prc_x1w)'}
	jret=lsi2nlg_calc(jobj)
	if debugTF:
		sys.stderr.write("==jret:\n{}\n".format(jret))
	return jret

def qs_split(xstr,d1='&',d2='='):
	""" split query string into a dict object """
	d = {k:v for k,v in [lx.split(d2) for lx in xstr.split(d1)]}
	return d

def qs_exec(xstr):
	""" convert commend string into a dict object """
	d = {}
	exec(xstr,globals(),d)
	return d

def qs_execOLD(xstr):
	""" convert commend string into a dict object """
	exec(xstr)
	del xstr
	d = locals()
	return d

def extra_opts(opts={},xkey='',method='JS',updTF=True):
	""" 
	update values of extraKey: xkey to DICT: opts 
	based on method: [JS|QS|XS]
	where
		JS: DICT/JSON type object
		QS: k1=v1;k2=v2;... query-like string
	"""	
	import ast	
	if xkey is None or len(xkey)<2:
		xkey = "extra{}".format(method)
	if xkey not in opts or opts[xkey] is None: 
		d = {}
	elif method == 'JS': # additional key/val like POST arguments
		d = ast.literal_eval(opts[xkey])
	elif method == 'QS': # additional key/val like GET arguments
		d = qs_split(opts[xkey])
	else: # additional key/val like python executable string
		d = qs_exec(opts[xkey])
	if updTF is True:
		opts.update(d)
		opts[xkey] = None
	return d

def generate_comment_hourly(ts,df,opts=None):
	import types
	import ast
	from _alan_calc import subDict
	from _alan_pppscf import vertex_locator,vertex_mnmx
	if opts is None and 'opt_csv2plot' in globals():
		opts, _ = opt_csv2plot([])
	if 'open' not in df and 'close' in df:
		df['open'] = df['close'].copy()
	if 'volume' not in df:
		df['volume'] = 0.0
	saveDB=getKeyVal(opts,'saveDB',True)
	debugTF=getKeyVal(opts,'debugTF',False)
	intradayTF=getKeyVal(opts,'intradayTF',False)
	dd = subDict(opts,['j2ts','j2name'],reverseTF=True)
	#if opts['extraJS'] is not None:
	#	try:
	#		dd.update(ast.literal_eval(opts['extraJS']))
	#	except Exception as e:
	#		print >> sys.stderr, str(e)
	if debugTF is True:
		sys.stderr.write("opts in generate_comment_hourly():\n{}\n".format(opts))
	if 'volstats' in dd and dd['volstats'] is not None:
		xmn,xsd = [float(x) for x in dd['volstats'].split(',') ]
		ubound,lbound = (xmn+xsd/2.0, xmn-xsd/2.0)
	else:
		ubound,lbound = 0,0
	dd.update(ubound=ubound,lbound=lbound)
	if 'rpt_hm' in dd:
		endTF = True if int(dd['rpt_hm'])>1530 else False
	else:
		endTF = False
	if df.size>0 and ts:
		dd.update(f=df)
		if 'close' in df:
			volS=dd['cvol'] if 'cvol' in dd else df['volume'].sum()*390/df.shape[0]
			dg, dh = vertex_locator(df['close'],**subDict(opts,['npar','debugTF']))
			#dh.reset_index(drop=True,inplace=True)
			if debugTF is True:
				sys.stderr.write("dh:\n{}\n".format(dh))
			prc_open = df['open'].iloc[0] if intradayTF is True else df['close'].iloc[0]
			# re-adjust cvol during the trading hours
			if endTF is True and 'cvol' in dd: 
				volS=dd['cvol']
			elif intradayTF is True: 
				volS = df['volume'].sum()*390/df.shape[0] 
			else:
				volS = df['volume'].iloc[-1]
			dk = vertex_mnmx(dh,prc_open,colx='actual')
			if debugTF is True:
				sys.stderr.write( "dMmMx:\n{}\n".format(dk))
			dd.update(dfdr=dh,dMnMx=dk,volS=volS,copen=prc_open)
			dd.update(fcsTrend=dh['sign'].iloc[-1]) # forthcoming trend 1:up,-1:down, 0: fair
		if debugTF is True:
			sys.stderr.write("==dd:\n{}\n".format(dd))
			sys.stderr.write("==Template ts:\n{}\n".format(ts))
		ret =  jj_fmt(ts,**dd)
		if debugTF is True:
			sys.stderr.write("==Comment:\n{}\n".format(ret))
		fname="hourly_headline_cn.j2"
		hlts = '{} include "{}" {}'.format('{%',fname,'%}')
		hlret =  jj_fmt(hlts,**dd)
		if debugTF is True:
			sys.stderr.write("==headline:\n{}\n".format(hlret))
		dd.update(mp3YN=True)
		mp3ret =  jj_fmt(ts,**dd)
		if debugTF is True:
			sys.stderr.write("==mp3Comment:\n{}\n".format(mp3ret))
		# FOR Technical Investment Strategy RUN
		techTF = getKeyVal(dd,'techTF',False)
		#if 'rpt_status' in dd and dd['rpt_status'] == 'trading': 
		if techTF is True:
			try:
				#pgDB =  conn2pgdb()
				from ohlc_minute_tst import find_stgy,run_narratives
				from _alan_calc import conn2pgdb,run_tech
				lang = dd['lang']
				if 'clop' not in df.columns:
					ma1=5;ma2=30
					mdf = run_tech(df, pcol='close',winLst=[ma1,ma2],debugTF=debugTF,nanTF=True)
				else:
					mdf=df.dropna(axis=0, how='any')
				minute_pnl = find_stgy(mdf,debugTF=debugTF)
				#retAddi = run_narratives_tst(dd['ticker'],mdf,minute_pnl,debugTF=debugTF)
				retAddi = run_narratives(dd['ticker'],mdf,minute_pnl,lang=lang,debugTF=debugTF,optx=dd)
				if debugTF is True:
					sys.stderr.write("==addiComment:\n{}\n".format(retAddi))
				ret += retAddi
				mp3ret += retAddi
				sys.stderr.write("==minute_pnl:\n{}\n".format(minute_pnl.tail().to_csv()))
			except Exception as e:
				sys.stderr.write( "**ERROR:{} @run_narratives :".format(str(e)) )
				sys.stderr.write( "ticker:{}\n".format(dd['ticker']) )
				sys.stderr.write( "mdf:\n{}\n".format(mdf.head().to_string()))
		tpLst = (types.BuiltinFunctionType,types.FunctionType,pd.DataFrame)
		jobj = { k:v for (k,v) in dd.items() if isinstance(v,tpLst) is False }
		clientM=None
		if opts['src'] in ['tw','hk','cn']:
			mdbname='ara_{}'.format(opts['src'].lower())
		else:
			mdbname='ara'

		jobj['comment'] = ret
		jobj['headline'] = hlret
		jobj['mp3comment'] = mp3ret
		sys.stderr.write("==ret:\n{}==hlret:\n{}\n".format( ret,hlret))
		if saveDB is True:
			for (k,v) in dd.items():
				if isinstance(v,pd.DataFrame) is True:
					jobj[k] = v.to_dict(orient='records')
			mobj,clientM,err_msg = write2mdb(jobj,clientM,dbname=mdbname,tablename='hourly_report',zpk={'label','rpt_date','rpt_hm'})
		else:
			for (k,v) in dd.items():
				if isinstance(v,pd.DataFrame) is True:
					jobj[k] = v
		if debugTF is True:
			sys.stderr.write( "==jobj in generate_comment_hourly():\n{}\n".format(jobj))
		return jobj
	else:
		return {}

def trend_forecast(ticker='^GSPC',freq='W',debugTF=False,pgDB=None,**optx):
	'''
	forecast daily/weekly/monthly as D/W/M forecasts of 'ticker'
	'''
	try:
		from _alan_ohlc_fcs import run_ohlc_fcs;
		(dd,dwm,datax)=run_ohlc_fcs(ticker,freq=freq,debugTF=debugTF,pgDB=pgDB,**optx)
		dc = dwm.query('freq=="{}"'.format(freq)).to_dict(orient='records')[0]
		dc.update(f=datax)
	except Exception as e:
		dc = dict(ticker=ticker,pos_pb=0,f={})
	return dc

def stock_screener(scrIds='most_actives',sortLst=['dollarValue'],ascLst=[False],**optx):
	'''
	get stock screened by 'scrIds'
	where 'scrIds' can be any of:
	  day_gainers,day_losers,most_actives,most_values
	  sector_gainers,sector_losers or customized 'scrIds'
	and return a dataframe up to 'nobs' rows with columns 'colx' default to:
	ticker, price, volume, change, changePercent, marketCap, dollarValue, sector, sector_cn, company, company_cn, pbdt
	sorted by columns 'sortLst' with ascending order as 'ascLst'
	'''
	scrIds= scrIds.lower()
	if scrIds=='day_gainers':
		df = x_screener(sortLst=['changePercent'],ascLst=[False],**optx)
	elif scrIds=='day_losers':
		df = x_screener(sortLst=['changePercent'],ascLst=[True],**optx)
	elif scrIds=='most_actives':
		df = x_screener(sortLst=['volume'],ascLst=[False],**optx)
	elif scrIds=='most_values':
		df = x_screener(sortLst=['dollarValue'],ascLst=[False],**optx)
	elif scrIds in ['5%','+5%']:
		optx.update(addiFilter='changePercent>=5&price>7.99',nobs=10)
		df = x_screener(sortLst=['changePercent'],ascLst=[False],**optx)
	elif scrIds=='-5%':
		optx.update(addiFilter='changePercent<-5&price>7.99',nobs=10)
		df = x_screener(sortLst=['changePercent'],ascLst=[True],**optx)
	elif scrIds=='sector_gainers':
		addiClause="and ticker in ('XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','XTL')"
		optx.update(addiFilter=0,nobs=11)
		df = x_screener(sortLst=['changePercent'],ascLst=[False],addiClause=addiClause,**optx)
	elif scrIds=='sector_losers':
		addiClause="and ticker in ('XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','XTL')"
		optx.update(addiFilter=0,nobs=11)
		df = x_screener(sortLst=['changePercent'],ascLst=[True],addiClause=addiClause,**optx)
	else:
		df = x_screener(sortLst=sortLst,ascLst=ascLst,**optx)
	return df

def x_screener(nobs=6,sortLst=['dollarValue'],ascLst=[False],dbname='ara',tablename='yh_qutoe_curr',addiFilter=1,debugTF=False,colx=[],addiClause='',**optx):
	'''
	support function to stock_screener() to get most active stocks 
	from MDB::dbname::tablename
	based on addiFilter criteria
	sorted by columns 'sortLst' with ascending order as 'ascLst'
	return a dataframe up to 'nobs' rows with columns 'colx' default to:
	ticker, price, volume, change, changePercent, marketCap, dollarValue, sector, sector_cn, company, company_cn, pbdt
	'''
	from _alan_calc import sqlQuery,subDF
	da,_,_=find_mdb({},tablename='yh_quote_curr',dbname=dbname,dfTF=True)
	da.dropna(subset=['close','volume'],inplace=True)
	if len(da)<1:
		return {}
	xqTmp="select * from mapping_ticker_cik where act_code=1"
	if len(addiClause)<1:
		addiClause="and sector not like '%%Index'"
	xqr="{} {}".format(xqTmp,addiClause)
	db = sqlQuery(xqr,dbname=dbname)
	da = da.merge(db[['ticker','company_cn','sector','sector_cn']],on='ticker')
	if len(da)<1:
		return {}
	da['price']=da['close']
	da['dollarValue']=(da['volume']*da['close']).astype(int)
	if addiFilter:
		if addiFilter==1:
			addiFilter='abs(changePercent)>0.5&price>7.99'
		elif addiFilter==2:
			addiFilter='abs(changePercent)>5&price>7.99'
		elif addiFilter==3:
			addiFilter='abs(changePercent)>2.5&price>4.99'
		elif isinstance(addiFilter,(int,float)):
			addiFilter='price>4.99'
		if debugTF:
			sys.stderr.write("==addiFilter:{}\n".format(addiFilter))
		try:
			da = da.query(addiFilter)
		except:
			pass
	if len(da)<1:
		return {}
	if len(sortLst)>0:
		df = da.sort_values(by=sortLst,ascending=ascLst)
	else:
		df = da
	df.reset_index(drop=True,inplace=True)
	df['ranking'] = [int(x)+1 for x in df.index.values]
	if colx is None or len(colx)<1:
		colx=['ranking','ticker','price','volume','change','changePercent','marketCap','dollarValue','Range52Week','pbdt','sector','sector_cn','company','company_cn']
	df=subDF(df,colx)
	if nobs>0:
		return df.iloc[:nobs]
	else:
		return df

if __name__ == '__main__':
	_alan_nlg(True)
