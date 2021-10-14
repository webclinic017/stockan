#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Program: lsi_daily.py
    Description: create ohlc commentary based on ohlc_hist and ohlc_pnl
    Input table required:
	prc_hist
	ohlc_pnl
	mapping_ticker_cik
	ohlc_latest_macd
	ohlc_ls_signal
	ohlc_pppscf
	ohlc_fcs 
	mapping_udf_comment_cn
	mapping_udf_comment_en
    Version: 0.66
    Output example: 
	蘋果08月16日每日快報：蘋果，代號A A P L，收盤價 $ 213.32。蘋果曾經在9天前出現符合烏雲蓋頂的反轉訊號，開始下降。目前，訊號所衍生的下降趨勢並未持續。 在過去12個月的價格囘側中，出現過23次烏雲蓋頂信號，獲利率為2%。 但是，MACD看漲的上昇訊號也在15天前出現，目前還在持續中並未改變，還在向上提升，可惜與烏雲蓋頂的方向並不一致。 此時趨勢呈現模糊不清的訊息，不建議採取任何進出行動。蘋果近期上升29個交易日, 總共為13%。 在過去14月裏其最大變幅在于07月06日和08月16日之間攀升13%。依據蘋果股價波動狀況，預估下一週價位有七成可能會在209到221元之間波動。上漲機率高達61%。
    Note:
	1. use MA(5) for confirm morning_star
	2. use MACD age and current level to confirm/conflict morning_star
    Function:
	def getdb_udfLst(category="stock",tbname="mapping_udf_comment",pgDB=None,lang="en"):
	def getdb_ticker(ticker,sqx,pgDB):
	def generate_daily_comment(f,ts=None,dotSign='.',prcn=0,usdSign='$',udfLst=None,lang="en"):
	def generate_comment_pricing(f,ts=None,dotSign='點',usdSign='元',accmm='',prcn=2,lang="cn",mp3YN=False):
	def generate_comment_title(f,ts=None,dotSign='點',usdSign='元',accmm='',prcn=2,lang="cn",mp3YN=False):
	def assign_pppscf_ts(lang="en"):
	def generate_comment_pppscf(f,dotSign='.',prcn=0,usdSign='$ ',udfLst=None,lang="en",ts=None):
	def run_daily_comment(aX,enhanceX=None,tempS=None,dotSign='.',prcn=0,usdSign='$',lang="cn"):
	def getdb_ohlc_daily(ticker,startYmd,endYmd,sector,label,pgDB):
	def assign_title_ts(lang):
	def assign_pricing_ts(lang):
	def run_comment_pricing(ticker='AAPL',label='',pgDB=None,dotSign='點',prcn=2,usdSign='元',lang="cn",mp3YN=False,ts_pricing=None,ts_title=None):
	def run_comment_pppscf(ticker='AAPL',label='',pgDB=None,dotSign='.',prcn=0,usdSign='$',lang="cn",fp=None,ts=None):
	def generate_comment_fcst_en(f,ts=None,dotSign='.',prcn=0,usdSign='$',udfLst=None,lang="en",mp3YN=False):
	def generate_comment_fcst_cn(f,ts=None,dotSign='.',prcn=0,usdSign='$',udfLst=None,lang="cn",mp3YN=False):
	def generate_cmt(f,ts=None,dotSign='.',prcn=0,usdSign='$',udfLst=None,lang="en",mp3YN=False,funcname=''):
	def run_comment_fcst(ticker='AAPL',label='',pgDB=None,dotSign='.',prcn=0,usdSign='',lang="cn",fp=None,ts=None,mp3YN=True):
	def save2mgdb_daily_comment(ret,dbM,tablename):
	def iteritem_daily_comment(j,tempS,tkX,mndt,mxdt,sector,label,daily_hdr,tablename,pgDB,mgDB,fp,wmode="replace",lang="cn",saveDB=True,mp3YN=False,dirname=None,debugTF=False):
	def assign_ts_ohlc(lang):
	def batch_daily_comment(tkLst=None,**kwargs):
	def opt_lsi_daily(argv,retParser=False):
    Last Mod., Fri Aug 17 14:47:09 EDT 2018
"""
from optparse import OptionParser
from _alan_str import *
from _alan_date import delta2dates
from _alan_calc import conn2pgdb,conn2mgdb,pqint
from bson import json_util
import pandas as pd
import os
import sys
if sys.version_info.major == 2:
	reload(sys)
	sys.setdefaultencoding('utf8')

def prn_dummy(du={}, fp=sys.stderr):
	""" print temp dict: du to file handle: fp
	"""
	if isinstance(fp,str) is True and len(fp)>0:
		fp = sys.stdout if fp == '-' else open(fp,'w')
	if isinstance(du,str) is True and len(du)>0:
			fp.write(du)
			fp.write("\n")
			return 1
	elif isinstance(du,dict) is False or len(du)<1:
		return None
	for k,v in  du.items():
		if isinstance(v,list) is True:
			for x in v:
				fp.write(x)
				fp.write("\n")
		elif isinstance(v,dict) is True:
			for j,x in v.items():
				fp.write(x)
				fp.write("\n")
		else:
			fp.write(v)
			fp.write("\n")

def assign_ts_title(lang):
	ts="{{label}}{{currDateWd}}每日快報：" if lang=="cn" else "{{label}} {{currDateWd}} Daily Alert: "
	return ts

def assign_ts_pricing(lang):
	ts="{{label}}，代號{{tickerWd}}，收盤價 {{priceWd}}。 " if lang=="cn" else "{{label}} ticker: {{tickerWd}}, closed at {{priceWd}}. "
	return ts

def assign_ts_pppscf(lang="en"):
	j2ts_cn="""{{label}}近期{{past_udf}}{{past_difday}}{{unitStr}}, 總共為{{past_act_chg_pct}}。 {%if act_chg_zscore > 1.96 %}
{# define syntax display #}在過去{{past_range}}裏其最大變幅在于{{latest_startYM}}和{{latest_endYM}}之間{{latest_udf}}{{latest_act_chg_pct}}。{% endif %}""" 
	j2ts_en="""{{label}} recently {{past_udf}} {{past_act_chg_pct}} for {{past_difday}} {{unitStr}}.
{# define syntax display #}{%if act_chg_zscore > 1.96 %}In the past {{past_range}}, the latest movement is between {{latest_startYM}} and {{latest_endYM}}, {{latest_udf}} {{latest_act_chg_pct}}.{% endif %}""" 
	return j2ts_en if lang=="en" else j2ts_cn

def assign_ts_ohlc(lang="en"):
	global tempStr_cn,macdStr_cn,macdEnhance_0_cn,macdEnhance_1_cn,tempStr_en,macdStr_en,macdEnhance_0_en,macdEnhance_1_en
	macdStr_en="""{{label}} {{pastAdv}} emerged the {{strategyName}} signal. So far, the {{comingTrendWd}} signal {{continueTrendWd}} and {{continueStatusClause}}. Backtest in the past {{monthWd}}，there are {{countWd}} of {{strategyName}} signals，{{trrAdv}} with total return at {{trrPct}}."""
	tempStr_en="""{{label}} {{pastAdv}} appeared a reversing signal {{strategyName}} {{pastDayWd}}, {{comingTrendWd}} started to emerge. Currently, the {{comingTrendWd}} signal {{continueTrendWd}} and {{continueStatusClause}}. Backtest in the past {{monthWd}}，there are {{countWd}} of {{strategyName}} signals，{{trrAdv}} with total return at {{trrPct}}. {{enhanceClause}} """
	macdEnhance_0_en="""{{prepositionWd}}，{{strategyName}} {{comingTrendWd}} also {{pastDayWd}} appeared，so far {{continueTrendWd}} {{continueStatusClause}}，its direction was not consistent with {{enhanceName}}. Those conflict singals do not suggest any action"""
	macdEnhance_1_en="""{{prepositionWd}}，{{strategyName}} {{comingTrendWd}} also {{pastDayWd}} appeared，so far {{continueTrendWd}} {{continueStatusClause}} and was consistent with {{enhanceName}}. Both singals enhanceed our suggestion."""
	macdStr_cn="""{{label}}{{pastAdv}}在{{pastDayWd}}出現符合{{strategyName}}的{{comingTrendWd}}訊號。目前，訊號所衍生的{{comingTrendWd}}趨勢{{continueTrendWd}}{{continueStatusClause}}。 在過去{{monthWd}}的價格囘側中，出現過{{countWd}}次{{strategyName}}信號，{{trrAdv}}獲利率為{{trrPct}}。 """
	tempStr_cn="""{{label}}{{pastAdv}}在{{pastDayWd}}出現符合{{strategyName}}的反轉訊號，開始{{comingTrendWd}}。目前，訊號所衍生的{{comingTrendWd}}趨勢{{continueTrendWd}}{{continueStatusClause}}。 在過去{{monthWd}}的價格囘側中，出現過{{countWd}}次{{strategyName}}信號，{{trrAdv}}獲利率為{{trrPct}}。 {{enhanceClause}}"""
	macdEnhance_0_cn="""{{prepositionWd}}，{{strategyName}}的{{comingTrendWd}}訊號也在{{pastDayWd}}出現，目前{{continueTrendWd}}{{continueStatusClause}}，可惜與{{enhanceName}}的方向並不一致。 此時趨勢呈現模糊不清的訊息，不建議採取任何進出行動。"""
	macdEnhance_1_cn="""{{prepositionWd}}，{{strategyName}}的{{comingTrendWd}}訊號也在{{pastDayWd}}出現，目前{{continueTrendWd}}{{continueStatusClause}}，與{{enhanceName}}的方向一致，進而發出更加明確的{{comingLookoutWd}}訊息。 在之前同一期間也有過{{countWd}}次{{strategyName}}，{{trrAdv}}獲利為{{trrPct}}{{trrAddClause}}。"""

	tempM={ "en":{"macd":macdStr_en,"temp":tempStr_en,"enhance":[macdEnhance_0_en,macdEnhance_1_en]},
		"cn":{"macd":macdStr_cn,"temp":tempStr_cn,"enhance":[macdEnhance_0_cn,macdEnhance_1_cn]}}
	#for lang in ["cn","en"]:
	daily_hdr=["ticker","name","price","trr","tsig","pnl_prd","curr_date","pbdate","lsc","curr_trend","curr_deriv","comment"]
	return (tempM[lang],daily_hdr)

def getdb_udfLst(category="stock",tbname="mapping_udf_comment",pgDB=None,lang="en"):
	try:
		if all( map(lambda x:x is not None,(tbname,pgDB)) ):
			tbname="{}_{}".format(tbname,lang) if lang != "en" else tbname
			sqx="SELECT * FROM {1} WHERE category={0!r} ORDER BY thd DESC"
			sqr=sqx.format(str(category),tbname)
			return pd.read_sql(sqr,pgDB)
		else:
			return None
	except Exception as e:
		pqint( "**ERROR @ getdb_udfLst():",str(e), file=sys.stderr)
		return None

def getdb_ticker(ticker,sqx,pgDB):
	try:
		gx=pd.read_sql(sqx.format(str(ticker)),pgDB)
	except:
		return None
	return gx

def generate_daily_comment(f,ts=None,dotSign='.',prcn=0,usdSign='$',udfLst=None,lang="en"):
	""" stock ohlc strategy comment for each strategy
	    essential f keys: ticker,name,price,trr,tsig,pbdate,curr_date,lsc,curr_trend,curr_deriv
	"""
	if ts is None:
		return ''
	try:
		enhanceClause= f['enhanceClause']
	except:
		enhanceClause= ""
	if 'enhanceName' in f: # For "enhance" case ONLY
		enhanceName = pattern_cn(f['enhanceName'],lang=lang)
		ynCode=f['ynCode']
	else:
		enhanceName=None
		ynCode=1
	xnDay=delta2dates(f['curr_date'],f['pbdate']) # to calc days of previous stragety
	bsFlg = True if f['lsc'] == "long" else False
	currFlg = True if f['curr_trend']*f['amount'] >=0 else False # check if current situation still persists
	currChg = True if f['curr_deriv'] >=0  else False # if situation persists, is it enhancing or diminishing
	pastAdv = ""
	trrAdv = ""
	trrAddClause = ""
	if lang == "cn":
		currDateWd=ymd2md(str(f['curr_date']),lang=lang)
		buySellVerb = "買入" if bsFlg else "賣出"
		prepositionWd = rdWord("另外|同時|不但如此") if ynCode==1 else rdWord("然而|但是|不過|可是")
		pastTrendWd = "跌" if bsFlg else "漲" # reverse of long/short for beforeward harami/star signal
		comingLookoutWd = rdWord("看漲|看多") if bsFlg else rdWord("看跌|看空")
		comingTrendWd = rdWord("上昇|看多|向上提升") if bsFlg else rdWord("下降|看空|下滑")
		continueTrendWd = rdWord("還在持續中並未改變|依然持續|仍在持續中") if currFlg else rdWord("並沒有持續|並未持續")
		if currFlg is True:
			upWd=rdWord("增强|升高|上升|向上提升")
			dnWd=rdWord("減弱|下降|滑落|下滑")
			continueStatusClause = "，還在"+upWd if currChg else "，但已逐漸"+dnWd
		else:
			continueStatusClause = ""
		if xnDay < 1:
			pastDayWd = "昨日"
		elif xnDay < 2:
			pastDayWd = "前一天".format(xnDay)
		else:
			pastDayWd = "{}天前".format(xnDay)
			pastAdv = "曾經"
		monthWd = "{}個月".format(f['pnl_prd'])
		negWd = "而" if f['trr'] >-0.005 else rdWord("只是|但是|可惜|可是")
		trrAdv = "" if f['trr'] >-0.005 else rdWord("只是|但是|可惜|可是")
		trrAddClause = "" if f['trr'] >-0.005 else rdWord("，並非有效策略|，策略有待商榷") 
	else :
		prepositionWd = rdWord("In the mean time|In addition") if ynCode==1 else rdWord("However|Yet|Nevertheless")
		currDateWd=ymd2md(f['curr_date'],lang=lang)
		buySellVerb = "buy" if bsFlg else "sell"
		pastTrendWd = "downtrend" if bsFlg else "uptrend" # reverse of long/short for beforeward harami/star signal
		comingTrendWd = "uptrend" if bsFlg else "downtrend"
		continueTrendWd = "still persists" if currFlg else "starts to reverse"
		continueStatusClause = "continues to increase" if currChg else "starts to reduce"
		if xnDay < 1:
			pastDayWd = "just"
		elif xnDay < 2:
			pastDayWd = "yesterday".format(xnDay)
		else:
			pastDayWd = "{} days ago".format(xnDay)
		monthWd = "{}-month".format(f['pnl_prd'])
		negWd = "" if f['trr'] >-0.005 else "however" 
	label = f['label'] if 'label' in f else ''
	ticker = f['ticker']
	if label is None or str(label)=='':
		label=ticker
	price = roundUSD(f['price'],1)
	strategyName = f['name']
	strategyName = pattern_cn(strategyName,lang=lang)
	countWd = f['tsig']
	trrPct = roundPct(f['trr'],0,pct=1)
	if trrPct == '0%' or trrPct == '-0%':
		trrPct = "持平而已" if lang == "cn" else "flat"
	pricingComm = ""
	dux=locals()
	ret=jj_fmt(ts,dux)
	#ret=ts.format(**dux)
	return(ret)

def generate_comment_pricing(f,ts=None,dotSign='點',usdSign='元',accmm='',prcn=2,lang="cn",mp3YN=False):
	""" convert essential key words: ticker,label,and price to comment
	"{label}代號{tickerWd}，收盤價 {priceWd}。"
	"""
	tickerWd=" ".join(list(str(f['ticker'])))
	label=f['label']
	prX=f['price']
	currDateWd=ymd2md(str(f['curr_date']),lang=lang)
	rfcn=roundMP3 if mp3YN is True else roundUSD
	priceWd=rfcn(prX,prcn,usdSign=usdSign,dotSign=dotSign)
	if lang=="cn" :
		tsWd='{}代號{}，收盤價 {}。' if ts is None else ts
	else:
		tsWd='{} ticker: {} closed at {}. ' if ts is None else ts
	if mp3YN is False:
		priceWd='{} {:{}.{}f}'.format('$',prX,',',prcn)
	dux=locals()
	ret=jj_fmt(tsWd,dux)
	#ret = tsWd.format(**dux)
	return ret

def generate_comment_title(f,ts=None,dotSign='點',usdSign='元',accmm='',prcn=2,lang="cn",mp3YN=False):
	""" generate daily title
	"{label}{currDateWd}每日快報："
	"""
	tsWd="{label}{currDateWd}每日快報：" if ts is None else ts
	currDateWd=ymd2md(str(f['curr_date']),lang=lang)
	label=f['label']
	dux=locals()
	ret=jj_fmt(tsWd,dux)
	#ret = tsWd.format(**dux)
	return ret

def generate_comment_pppscf(f,dotSign='.',prcn=0,usdSign='$ ',udfLst=None,lang="en",ts=None):
	""" stock past performance comment
	"""
	if ts is None:
		ts = assign_ts_pppscf(lang=lang)
	fp=f.iloc[-1].to_dict()
	#for ky,va in fp.items():
	#	exec("{}=va".format(ky)) 
	unitStr=fq2unit_str(fp['freq'],lang)
	past_udf=udfStr(fp['act_chg_zscore'],udf=None,zs=0.0001,lang=lang)
	past_difday=fp['difday']
	past_act_chg_pct=roundPct(fp['act_chg'],n=0,pct=100,dotSign=dotSign,lang=lang)
	past_range=str(delta2dates(int(f.iloc[-1]['pbdate']),int(f.iloc[0]['pbdate']),fq='M')) + ("-month" if lang=="en" else "月")
	fx=f.query("sig_act_chg==1").iloc[0].to_dict()
	#for ky,va in fp.items():
	#	exec("{}=va".format(ky)) 
	act_chg_zscore=fx.pop('act_chg_zscore',0)
	act_chg=fx.pop('act_chg',0)
	xdate=fx.pop('xdate',0)
	pdate=fx.pop('pdate',0)
	latest_startYM=ymd2md(xdate,ymd='%Y-%m-%d',ym="%B %Y",lang=lang)
	latest_endYM=ymd2md(pdate,ymd='%Y-%m-%d',ym="%B %Y",lang=lang)
	latest_udf=udfStr(act_chg_zscore,udf=None,zs=0.0001,lang=lang)
	latest_act_chg_pct=roundPct(act_chg,n=0,pct=100,dotSign=dotSign,lang=lang)
	dux=locals()
	ret=jj_fmt(ts,dux)
	global gData
	gData=dux.copy()
	del gData['fp']
	del gData['fx']
	return ret

def run_daily_comment(aX,enhanceX=None,tempS=None,dotSign='.',prcn=0,usdSign='$',lang="cn"):
	if tempS is None:
		return None
	if aX is None and enhanceX is not None:
		aX = enhanceX
		enhanceX = None
		tempX=tempS["macd"]
	else:
		tempX=tempS["temp"]

	if enhanceX is not None:
		enhanceX['enhanceName']=str(aX['name'])
		ynCode = 1 if aX['lsc'] == enhanceX['lsc'] else 0 # check if both strategies for long/short are the same
		tempE=tempS['enhance'][ynCode]
		enhanceX['ynCode']=ynCode
		enhanceClause= generate_daily_comment(enhanceX,tempE,dotSign=dotSign,lang=lang)
		aX['enhanceClause']=enhanceClause
	else:
		aX['enhanceClause']=""

	ret=generate_daily_comment(aX,tempX,dotSign=dotSign,lang=lang)
	aX['nsig']=aX['tsig']
	aX['comment_ohlc']=ret
	return aX

def getdb_ohlc_daily(ticker,startYmd,endYmd,sector,label,pgDB):
	""" get info from ohlc_hist and ohlc_pnl for ohlc strategy info
	"""
	sq1="SELECT * FROM ohlc_latest_macd WHERE ticker = {0!r} order by pbdate DESC"
	sq2="SELECT * FROM ohlc_ls_signal WHERE ticker = {0!r} order by pbdate DESC"
	#sq1="SELECT ticker,pbdate,price,ma5,ma20,price-ma5 as ma5trend,signal_value_macd,signal_buysell_macd FROM ohlc_hist WHERE ticker = {0!r} and pbdate >= (SELECT max(pbdate) FROM ohlc_hist WHERE ticker = {0!r} and signal_buysell_macd <> 0 ) order by pbdate DESC"
	#sq2="select distinct * from ohlc_pnl where ticker={0!r} and sig=1 and lsc<>'combo' and name similar to '%%(star|harami|MACD)' order by name,lsc,pbdate DESC"
	g1=pd.read_sql(sq1.format(ticker),pgDB)
	g2=pd.read_sql(sq2.format(ticker),pgDB).loc[:,["ticker","name","pbdate","price","lsc","amount","cum_amounts","sig","tsig","ttr_hold","trr"]]
	pnl_prd=delta2dates(endYmd,startYmd,fq="M")
	pbdate=endYmd
	try:
		m1=g2.loc[~g2['name'].str.contains("MACD")].reset_index(drop=True).sort_values(by=['pbdate'],ascending=False).iloc[0]
		m1['curr_trend'] = m1['curr_deriv'] = 0
		if len(g1)>0:
			m1['curr_trend']=g1['ma5trend'][0] 
		if len(g1)>1:
			m1['curr_deriv'] = 1 if abs(g1['ma5trend'][0])>=abs(g1['ma5trend'][1]) else -1
		m1['curr_date']=pbdate
		m1['pnl_prd']=pnl_prd
		m1['sector']=sector
		m1['label']=label
	except:
		m1=None
	try:
		m2=g2.loc[g2['name'].str.contains("MACD")].reset_index(drop=True).sort_values(by=['pbdate'],ascending=False).iloc[0]
		m2['curr_trend'] = m2['curr_deriv'] = 0
		if len(g1)>0:
			m2['curr_trend'] = g1['signal_value_macd'][0]
		if len(g1)>1:
			m2['curr_deriv'] = 1 if abs(g1['signal_value_macd'][0])>=abs(g1['signal_value_macd'][1]) else -1
		m2['curr_date']=pbdate
		m2['pnl_prd']=pnl_prd
		m2['sector']=sector
		m2['label']=label
	except:
		m2=None
	return(pbdate,m1,m2)

def run_comment_pricing(ticker='AAPL',label='',pgDB=None,dotSign='點',prcn=2,usdSign='元',lang="cn",mp3YN=False,ts_pricing=None,ts_title=None):
	if ts_title is None:
		ts_title=assign_ts_title(lang)
	if ts_pricing is None:
		ts_pricing=assign_ts_pricing(lang)
	sqx="SELECT name as ticker, pbdate as curr_date,close as price,volume FROM prc_hist WHERE name={0!r} ORDER BY pbdate DESC limit 1"
	fp=getdb_ticker(ticker,sqx,pgDB).iloc[0]
	if 'label' not in fp:
		fp['label']=label
	try:
		comment_pricing = generate_comment_pricing(fp,ts=ts_pricing,dotSign=dotSign,prcn=prcn,usdSign=usdSign,lang=lang,mp3YN=mp3YN)
		comment_title = generate_comment_title(fp,ts=ts_title,dotSign=dotSign,prcn=prcn,usdSign=usdSign,lang=lang,mp3YN=mp3YN)
		pqint( "comment_pricing:",comment_pricing, file=sys.stderr)
		pqint( "comment_title:",comment_title, file=sys.stderr)
	except Exception as e:
		pqint( "**ERROR @ run_comment_pricing():",str(e), file=sys.stderr)
		return None
	return (comment_pricing,comment_title)

def run_comment_pppscf(ticker='AAPL',label='',pgDB=None,dotSign='.',prcn=0,usdSign='$',lang="cn",fp=None,ts=None):
	if fp is None:
		sqx="SELECT * FROM ohlc_pppscf WHERE ticker={0!r} ORDER BY pbdate"
		fp=getdb_ticker(ticker,sqx,pgDB)
	if 'label' not in fp:
		fp['label']=label
	if 'freq' not in fp:
		fp['freq']='D'
	if 'category' not in fp:
		fp['category']='stock'
	category=fp['category'].iloc[-1]

	#====Performance Comment====================
	udfLst=getdb_udfLst(category,"mapping_udf_comment",pgDB=pgDB,lang=lang)
	try:
		ret = generate_comment_pppscf(fp,dotSign=dotSign,prcn=prcn,usdSign=usdSign,udfLst=udfLst,lang=lang,ts=ts)
	except Exception as e:
		pqint( "**ERROR @ run_comment_pppscf():",str(e), file=sys.stderr)
		#pqint( fp, file=sys.stderr)
		return None
	return ret

def generate_comment_fcst_en(f,ts=None,dotSign='.',prcn=0,usdSign='$',udfLst=None,lang="en",mp3YN=False):
	""" stock weekly forecast comment
		required f.fields: low_bound,up_bound,prc_cur,prc_x1w,prc_fcs,sigma,ticker,label,
		required ts fields: label,lowerWd,upperWd,posPbWd
	"""
	if ts is None:
		ts="""{label} closed {xdChgWd} at {price}, {xwChgWd} for the week. The {xwChgWd} movement is {cmpWd} the historical volatility of {sigmaWd}. Our {label} forecast for the next week is {nwTrendWd} with a probability of {posPbWd} and {plevelWd} chance of closing {rangeWd}."""
	#for (ky,va) in f.items(): 
	#	exec("{}=va".format(ky))
	low_bound,up_bound,prc_cur,prc_x1w,prc_fcs,xchg1d =\
		f['low_bound'],f['up_bound'],f['prc_cur'],f['prc_x1w'],f['prc_fcs'],f['xchg1d']
	prc_chg,rrate_sigma,plevel,pos_pb = f['prc_chg'],f['rrate_sigma'],f['plevel'],f['pos_pb']
	category = f['category']
	prc_dif=prc_cur-prc_x1w
	rfcn=roundMP3 if mp3YN is True else roundUSD
	rfcnPct=roundPctMP3 if mp3YN is True else roundPct
	unitStr=fq2unit_str(f['freq'],lang) if 'freq' in f else "週"
	xdTrendWd=udfStr(xchg1d,["up","down","unchanged"],0.005)
	xdChgWd=rfcn(xchg1d,2,usdSign=usdSign,dotSign=dotSign)
	xdayWd=rdWord("the previous trading day|previously|the last trading day")
	movingWd=rdWord("movement|change|difference")
	if xdTrendWd == "unchanged":
		xdChgWd = ''
	xwTrendWd=udfStr(prc_chg,["an increase","a decrease","unchanged"],0.0005)
	xwChgWd=rfcnPct(prc_chg,n=1,pct=100,dotSign=dotSign)
	xwDifWd=rfcn(prc_dif,2,usdSign=usdSign,dotSign=dotSign) 
	xwBpsWd=rfcn(prc_dif*100,0,usdSign=usdSign,dotSign=dotSign) 
	if xwTrendWd == "unchanged":
		xwChgWd = ''
	cmpWd=udfStr(abs(prc_chg)-rrate_sigma,["more than","less than","about the same as"],0.005)
	plevelWd=roundPct(plevel,n=0,pct=100)
	sigmaWd=rfcnPct(rrate_sigma,n=1,pct=100,dotSign=dotSign)
	posPbWd=roundPct(pos_pb,n=0,pct=100)
	nwTrendWd = "up"

	xt1wZ=prc_chg/rrate_sigma*100
	rt1w=f['prc_fcs']/f['prc_cur']
	rt1wZ=rt1w/f['sigma']*100
	pastTrendWd = performance_verb(xt1wZ,udfLst)
	comingTrendWd = performance_verb(rt1wZ,udfLst)

	vntDateWd=ymd2md(str(vntdate),ym="%B %d",lang=lang) if 'vntdate' in f else ""
	label = f['label'] if 'label' in f else ''
	ticker = f['ticker']
	category = f['category'] if 'category' in f else 'stock'
	unitSign='元' if lang=='cn' and category=='stock' else ''
	price = rfcn(f['prc_cur'],prcn,usdSign=usdSign,dotSign=dotSign)
	pn=0 if low_bound>99.4 else prcn
	lowerWd = rfcn(f['low_bound'],pn,usdSign=usdSign,dotSign=dotSign)
	upperWd = rfcn(f['up_bound'],pn,usdSign=usdSign,dotSign=dotSign)
	
	if label is None or str(label)=='':
		label=ticker
	rangeWd=btwStr(lowerWd,upperWd,["between","and"])

	dux=locals()
	ret=jj_fmt(ts,dux)
	for x,y in zip(["xdChgWd","xwChgWd","xwDifWd","xwBpsWd","sigmaWd","price","lowerWd","upperWd"],[xdChgWd,xwChgWd,xwDifWd,xwBpsWd,sigmaWd,price,lowerWd,upperWd]):
		dux.update({x:num2MP3(y,lang=lang)})
	ret_mp3=jj_fmt(ts,dux)
	global gData
	gData=dux.copy()
	del gData['f']
	return(ret,ret_mp3)

def generate_comment_fcst_cn(f,ts=None,dotSign='.',prcn=0,usdSign='$',udfLst=None,lang="cn",mp3YN=False):
	""" stock weekly forecast comment
		required f.fields: low_bound,up_bound,prc_cur,prc_x1w,prc_fcs,sigma,ticker,label,
		required ts fields: label,lowerWd,upperWd,posPbWd
	"""
	freq="D"
	#for (ky,va) in f.items(): 
	#	exec("{}=va".format(ky))
	low_bound,up_bound,prc_cur,prc_x1w,prc_fcs,xchg1d =\
		f['low_bound'],f['up_bound'],f['prc_cur'],f['prc_x1w'],f['prc_fcs'],f['xchg1d']
	prc_chg,rrate_sigma,plevel,pos_pb = f['prc_chg'],f['rrate_sigma'],f['plevel'],f['pos_pb']
	category = f['category']
	prc_dif=prc_cur-prc_x1w
	rfcn=roundMP3 if mp3YN is True else roundUSD
	rfcnPct=roundPctMP3 if mp3YN is True else roundPct
	unitStr=fq2unit_str(f['freq'],lang) if 'freq' in f else "週"
	xdTrendWd=udfStr(xchg1d,None,0.005,lang=lang)
	xdChgWd=rfcn(xchg1d,2,usdSign=usdSign,dotSign=dotSign,lang=lang)
	xdayWd=rdWord("前一個交易日|前一天") if freq=="D" else rdWord("上一|前一")+unitStr
	movingWd=rdWord("變化|改變")
	if abs(xchg1d) <= 0.005:
		xdChgWd = ''
	xwTrendWd=udfStr(prc_chg,None,0.005,lang=lang)
	xwChgWd=rfcnPct(prc_chg,n=1,pct=100,dotSign=dotSign,lang=lang)
	xwDifWd=rfcn(prc_dif,2,usdSign=usdSign,dotSign=dotSign) 
	xwBpsWd=rfcn(prc_dif*100,0,usdSign=usdSign,dotSign=dotSign) 
	if abs(prc_chg) <= 0.0005:
		xwChgWd = ''
	cmpWd=udfStr(abs(prc_chg)-rrate_sigma,["大於","小於","相當於"],0.005)
	plevelWd=roundPct(plevel,n=0,pct=100)
	sigmaWd=rfcnPct(rrate_sigma,n=1,pct=100,dotSign=dotSign,lang=lang)
	posPbWd=roundPct(pos_pb,n=0,pct=100)
	nwTrendWd = "上升"

	xt1wZ=prc_chg/rrate_sigma*(1 if category=='stock' else 1)
	rt1w=f['prc_fcs']/f['prc_cur'] - 1
	rt1wZ=rt1w/f['sigma']*(1 if category=='stock' else 1)
	#xsign = 1 if ticker[-2:] == "US" else -1
	xsign = -1 # cn version always use EURO direction
	if category == 'currency':
		pastTrendWd = performance_verb(xt1wZ*xsign,udfLst)
		comingTrendWd = performance_verb(rt1wZ*xsign,udfLst)
	else:
		pastTrendWd = performance_verb(xt1wZ,udfLst)
		comingTrendWd = performance_verb(rt1wZ,udfLst)

	vntDateWd=ymd2md(str(vntdate),ym="%B %d",lang=lang) if 'vntdate' in f else ""
	label = f['label'] if 'label' in f else ''
	ticker = f['ticker']
	category = f['category'] if 'category' in f else 'stock'
	unitSign='元' if lang=='cn' and category=='stock' else ''
	price = rfcn(f['prc_cur'],prcn,usdSign=usdSign,dotSign=dotSign)
	pn=0 if low_bound>99.4 else prcn
	lowerWd = rfcn(f['low_bound'],pn,usdSign=usdSign,dotSign=dotSign)
	upperWd = rfcn(f['up_bound'],pn,usdSign=usdSign,dotSign=dotSign)
	
	if label is None or str(label)=='':
		label=ticker
	if 'pos_pb' in f:
		pp=f['pos_pb']*100.
		ppAt="大約為" if pp>=49.5 else "只達到"
		ppAt="高達" if pp>=59.5 else ppAt
		ppUp="貶值" if xsign==1 else "升值"
		ppUp="上漲" if category!="currency" else ppUp
		posPbWd="{}機率{}{:.0f}%。".format(ppUp,ppAt,pp)
	if lowerWd == upperWd:
		rangeWd = "變化不大，只會在{}{}內上下稍稍波動".format(lowerWd,unitSign) 
	else:
		rangeWd = "會在{}到{}{}之間波動".format(lowerWd,upperWd,unitSign) 
	dux=locals()
	ret=jj_fmt(ts,dux)
	for x,y in zip(["xdChgWd","xwChgWd","xwDifWd","xwBpsWd","sigmaWd","price","lowerWd","upperWd"],[xdChgWd,xwChgWd,xwDifWd,xwBpsWd,sigmaWd,price,lowerWd,upperWd]):
		dux.update({x:num2MP3(y,lang=lang)})
	ret_mp3=jj_fmt(ts,dux)
	global gData
	gData=dux.copy()
	del gData['f']
	return(ret,ret_mp3)

def generate_cmt(f,ts=None,dotSign='.',prcn=0,usdSign='$',udfLst=None,lang="en",mp3YN=False,funcname=''):
	try:
		funcN="{}_{}".format(funcname,lang)
		funcArg=globals()[funcN]
		xcmt=funcArg(f,ts=ts,dotSign=dotSign,prcn=prcn,usdSign=usdSign,udfLst=udfLst,mp3YN=mp3YN)
	except Exception as e:
		pqint( "**ERROR @ {}():{}".format("generate_cmt",str(e)), file=sys.stderr)
		xcmt=''
	return xcmt

def run_comment_fcst(ticker='AAPL',label='',pgDB=None,dotSign='.',prcn=0,usdSign='',lang="cn",fp=None,ts=None,mp3YN=True):
	if fp is None:
		#sqx="SELECT * FROM ara_weekly_forecast WHERE ticker={0!r} and freq='W'"
		sqx="SELECT * FROM ohlc_fcs WHERE ticker={0!r} and freq='W'"
		try:
			fp=getdb_ticker(ticker,sqx,pgDB).iloc[0]
		except:
			pqint( "No forecast available!", file=sys.stderr)
			return ''
	if 'label' not in fp:
		fp['label']=label
	if 'category' not in fp:
		fp['category']='stock'
	category=fp['category']
	
	if ts is None and lang=="cn":
		#ts="""{label}在前一週{pastTrendWd}，目前收盤價{price}元，我們預期在下一週將{comingTrendWd}，介于{lowerWd}至{upperWd}元之間。"""
		if category=='stock':
			ts="""依據{label}股價波動狀況，預估下一週價位有七成可能{rangeWd}。{posPbWd}"""
		else:
			ts="""依據{label}波動狀況，預估下一{unitStr}有七成可能{rangeWd}。{posPbWd}"""

	#====Performance Comment====================
	udfLst=getdb_udfLst(category,"mapping_udf_comment",pgDB=pgDB,lang=lang)
	ret,rmp3 = generate_cmt(fp,ts=ts,dotSign=dotSign,prcn=prcn,usdSign=usdSign,udfLst=udfLst,lang=lang,funcname="generate_comment_fcst",mp3YN=mp3YN)
	return ret

def save2mgdb_daily_comment(ret,dbM,tablename):
	if dbM is None:
		return None
	pqint( ret, file=sys.stderr)
	mobj=json_util.loads(ret.to_json())
	dbM[tablename].delete_one({"ticker":mobj["ticker"],"curr_date":mobj["curr_date"]})
	dbM[tablename].insert_one(mobj)
	return len(mobj)

def iteritem_daily_comment(j,tempS,tkX,mndt,mxdt,sector,label,daily_hdr,tablename,pgDB,mgDB,fp,wmode="replace",lang="cn",saveDB=True,mp3YN=False,dirname=None,debugTF=False):
	
	if mp3YN is True:
		dotSign = "點" if lang== "cn" else "point"
	else:
		dotSign = "."
			
	pqint("------",j,tkX,mndt,mxdt,sector,label, file=sys.stderr)
	try:
		(comment_pricing,comment_title)=run_comment_pricing(tkX,label,pgDB,dotSign=dotSign,lang=lang,mp3YN=mp3YN)
	except Exception as e:
		pqint( "**ERROR {}. {} @ {}: {}".format(j,tkX,"run_comment_pricing",str(e)), file=sys.stderr)
		comment_pricing=''
		comment_title=''
	try:
		comment_pppscf=run_comment_pppscf(tkX,label,pgDB,dotSign=dotSign,lang=lang)
		pqint( "{0}:{1}".format("comment_pppscf",comment_pppscf) , file=sys.stderr)
	except Exception as e:
		pqint( "**ERROR {}. {} @ {}: {}".format(j,tkX,"run_comment_pppscf",str(e)), file=sys.stderr)
		comment_pppscf=''
	try:
		comment_fcst=run_comment_fcst(tkX,label,pgDB,dotSign=dotSign,lang=lang,mp3YN=mp3YN)
		pqint( "{0}:{1}".format("comment_fcst",comment_fcst) , file=sys.stderr)
	except Exception as e:
		pqint( "**ERROR {}. {} @ {}: {}".format(j,tkX,"run_comment_fcst",str(e)), file=sys.stderr)
		comment_fcst=''
	try:
		(pbdate,m1,m2)=getdb_ohlc_daily(str(tkX),mndt,mxdt,sector,label,pgDB)
		aX=run_daily_comment(m1,m2,tempS,dotSign=dotSign,lang=lang)
		pqint( "{0}:{1}".format("comment_ohlc",aX['comment_ohlc']) , file=sys.stderr)
		# pqint(aX, file=sys.stderr)
	except Exception as e:
		pqint( "**ERROR {}. {} @ {}: {}".format(j,tkX,"run_daily_comment",str(e)), file=sys.stderr)
		return None

	aX['comment_pppscf']=comment_pppscf
	aX['comment_fcst']=comment_fcst
	aX['comment_pricing']=comment_pricing
	aX['comment_title']=comment_title
	comment='{comment_title}{comment_pricing}{comment_ohlc}{comment_pppscf}{comment_fcst}'.format(**aX)
	aX['comment']=comment
	pqint( "{0}:\n{1}".format("COMMENT",comment), file=sys.stderr)
	pqint( "{}".format(comment), file=sys.stdout)
	if debugTF:
		pqint( aX, file=sys.stderr)

	#= re-assign aX to tmpx and save to DB:tablename
	daily_hdr= daily_hdr + ["comment_title","comment_pppscf","comment_fcst","comment_pricing","comment_ohlc"]
	tmpx=aX[daily_hdr]

	if saveDB is True:
		#- save tmpx to database pgDB
		tmpx.to_frame().T.to_sql(tablename,pgDB,index=False,schema="public",if_exists=wmode)
		#- save tmpx to database mgDB
		save2mgdb_daily_comment(tmpx,mgDB,tablename)
		if debugTF is True:
			pqint( "save to [{}]".format(tablename), file=sys.stderr)
	else:
		if debugTF is True:
			pqint( "no save to table:[{}]".format(tablename), file=sys.stderr)

	#- write comment to file fdaily
	if dirname is not None and os.path.exists(dirname) is True:
		fdailyname="{}/daily_{}_{}_{}.txt".format(dirname,lang,aX['ticker'],aX['curr_date'])
		fdaily=open(fdailyname,"w")
		fdaily.write(aX['comment'])
		fdaily.close()
	#- write tmpx to file fp
	xstr="|".join(["{}"]*len(daily_hdr))+"\n"
	if fp is None:
		return tmpx
	if(j==0):
		fp.write(xstr.format(*daily_hdr))
	fp.write(xstr.format(*tmpx) )
	return tmpx

def batch_daily_comment(tkLst=None,**kwargs):
	for ky,va in kwargs.items():
		exec("{}=va".format(ky))
	pgDB=conn2pgdb(dbname=dbname,hostname=hostname)
	mgDB=conn2mgdb(dbname=dbname.replace('.','_'),hostname=hostname)
	#====OHLC Comment===========================
	pqint( "====OHLC Comment===========================", file=sys.stderr)
	if tablename is None:
		tablename="{}_{}".format("ohlc_daily_comment",lang)
		if mp3YN is True:
			tablename="{}_{}".format(tablename,"mp3")
	else:
		tablename=tablename

	# Initial assignments
	(tempS,daily_hdr)=assign_ts_ohlc(lang=lang)
	if filename is None:
		fp = None
	elif filename == "stdout" or tablename == "-":
		fp = sys.stdout
	elif filename == "stderr":
		fp = sys.stderr
	else:
		fp = open(filename,"w") 

	# Get batch list
	sqr="SELECT s.*,m.sector,m.company{} as company FROM mapping_ticker_cik m right join (select ticker,min(pbdate) as mndt,max(pbdate) as mxdt from ohlc_pnl group by ticker) as s ON m.ticker=s.ticker order by s.ticker".format("" if lang=="en" else "_"+lang)
	dataM=pd.read_sql(sqr,pgDB)
	#dataM=pd.read_sql(sqr,pgDB)[9:10]
	if tkLst is not None:
		dx = dataM[dataM['ticker'].isin(tkLst)]
		if dx.empty is True:
			pqint("{} not found!".format(tkLst), file=sys.stderr)
			return None	
		else:
			dataM = dx.reset_index(drop=True)

	pqint("START batch_daily_comment-------------------------------------------", file=sys.stderr)
	for j, tRw in dataM.iterrows():
		(tkX,mndt,mxdt,sector,label)=tRw.values
		iteritem_daily_comment(j,tempS,tkX,mndt,mxdt,sector,label,daily_hdr,tablename,pgDB,mgDB,fp,wmode=wmode,lang=lang,saveDB=saveDB,mp3YN=mp3YN,dirname=dirname,debugTF=debugTF)
		wmode='append'
	#fp.close()
	pqint("END batch_daily_comment-------------------------------------------", file=sys.stderr)
	print(kwargs.keys())
	print(kwargs.values())

def opt_lsi_daily(argv,retParser=False):
	""" command-line options initial setup
	    Arguments:
		argv:   list arguments, usually passed from sys.argv
		retParser:      OptionParser class return flag, default to False
	    Return: (options, args) tuple if retParser is False else OptionParser class
	"""
	parser = OptionParser(usage="usage: %prog [option] SYMBOL1 ...", version="%prog 0.66",
		description="Create ohlc commentary based on ohlc_hist and ohlc_pnl")
	parser.add_option("-d","--database",action="store",dest="dbname",default="ara",
		help="database name (default: eSTAR_2)")
	parser.add_option("","--host",action="store",dest="hostname",default="localhost",
		help="db host name (default: localhost)")
	parser.add_option("-t","--table",action="store",dest="tablename",
		help="db tablename (default: None)")
	parser.add_option("","--file",action="store",dest="filename",
		help="db filename (default: None)")
	parser.add_option("","--dirname",action="store",dest="dirname",
		help="db dirname to save comment of each ticker (default: None)")
	parser.add_option("-w","--wmode",action="store",dest="wmode",default="replace",
		help="db table write-mode [replace|append|fail] (default: replace)")
	parser.add_option("-l","--lang",action="store",dest="lang",default="cn",
		help="db language mode [cn|en] (default: cn)")
	parser.add_option("","--no_database_save",action="store_false",dest="saveDB",default=True,
		help="no save to database (default: save to database)")
	parser.add_option("","--use_mp3",action="store_true",dest="mp3YN",default=False,
		help="comment use mp3 style")
	parser.add_option("","--debug",action="store_true",dest="debugTF",default=False,
		help="debugging (default: False)")
	(options, args) = parser.parse_args(argv[1:])
	if retParser is True:
		return parser
	return (vars(options), args)

if __name__ == "__main__" :
	(options, args)=opt_lsi_daily(sys.argv)
	tkLst=None
	if len(args)>0:
		if args[0]=='-':
			tkLst = sys.stdin.read().strip().split("\n") 
		else:
			tkLst=args 
			options['saveDB']=False
	batch_daily_comment(tkLst,**options)
