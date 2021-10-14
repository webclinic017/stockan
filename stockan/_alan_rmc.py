#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Remote Model Calculation

last mod., Tue Oct 27 17:30:23 EDT 2020
'''
import sys
#sys.path.append("/apps/fafa/pyx/tst/")
#from __future__ import print_function
import pandas as pd, numpy as np, json, os, datetime
import simplejson
from pandas import Timestamp
from bson.objectid import ObjectId
from flask import render_template,jsonify,Response,make_response
#from jinja2 import Template as jjTemplate
from types import *
from subprocess import Popen, PIPE
from glob import glob, iglob
from _alan_str import lsi2nlg_calc,combine_cmd2dict,jj_fmt,sysCall,find_mdb,write2mdb
from _alan_calc import subDict,sqlQuery,getKeyVal,pqint,subDF,subDict,renameDict
from _alan_date import tg_latest2week,tg_next2week,next_date,dt2ymd,ymd2dt,s2dt
from alanapi import run_alanapi,opt_alanapi,search_comment,search_factor,search_quote,search_hist,search_history,search_list,search_allocation,search_mp3,search_ohlc
#,udfStr,roundUSD
from _alan_pppscf import batch_pppscf
from _alan_ohlc_fcs import batch_ohlc_fcs
from math import *
import requests
from importlib import import_module
if sys.version_info.major == 2:
	reload(sys)
	sys.setdefaultencoding('utf8')
pd.options.display.float_format='{:,.2f}'.format

def dtCvt (x):
	if isinstance(x, (ObjectId,datetime.datetime,Timestamp)):
		return x.__str__()
	return x

def jsonCvt(x): return x.__str__() if hasattr(x,'__str__') else x

#dtCvt = lambda x: str(x) if isinstance(x, (ObjectId)) else x
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:,g}'.format

def pqrint(*args,**kwargs):
	return pqint(*args,**kwargs)

def rPost(url='',data={},timeout=5,ret='{}'):
	try:
		if len(url)<0:
			return ret
		r = requests.post(url, data=data,timeout=timeout)
		r.raise_for_status()
		if isinstance(r,dict):
			ret=r
		else:
			ret=r.content
	except requests.exceptions.HTTPError as e:
		sys.stderr.write("{}\n".format(e.response.text))
	return ret

def err_page(htmlName='errorhandler.html',code=-500,name="Invalid Calculation",description=''):
	''' return error message via html page
	'''
	return render_template(htmlName,code=code,name=name,description=description)

def display_page(pathname='index',request=None,**optx):
	return show_page(pathname=pathname,request=request,**optx)

def show_page(pathname='index',request=None,**optx):
	sys.stderr.write("===START show_page request:{}\n".format(request))
	opts=wrap_request(request)
	sys.stderr.write(" --opts:{}\n".format(opts))
	try:
		pageBase=pathname
		funcName='page_{}'.format(pageBase)
		htmlName=opts.pop('html',None)

		#----------------------------------------------------------------
		if funcName=='page_api' and htmlName is None:
			# Run 'page_api' command
			dd = page_api(**opts)
			if isinstance(dd,str):
				#return dd
				ret=Response(dd,mimetype='application/json')
				return ret
			ret=Response(simplejson.dumps(dd,ignore_nan=True),mimetype='application/json')
			return ret
		elif funcName in globals() and hasattr(globals()[funcName],'__call__'):
			# Run corresponding funcName like page_???()
			pass
		elif 'templates/{}'.format(htmlName) in glob('templates/*html'):
			# Newly added @ Sat Oct 24 21:25:43 EDT 2020
			# Process corresponding html page without funcName like page_???()
			ret = process_page(htmlName, **opts)
			return ret
		else:
			# Run home page page_index() as default
			sys.stderr.write("**WARNINGS:{} not found".format(funcName))
			funcName='page_{}'.format('index')
			sys.stderr.write(", use:{} instead".format(funcName))

		#----------------------------------------------------------------
		funcArg=globals()[funcName]

		sys.stderr.write(" --pathname:{}\n".format(pathname))
		sys.stderr.write(" --funcName:{}\n".format(funcName))
		sys.stderr.write(" --FuncObj:{}\n".format(funcArg))
		sys.stderr.write(" --show_page {} INPUT/opts:\n{}\n".format(funcName,opts))

		#----------------------------------------------------------------
		# Run corresponding command page_???
		dd = funcArg(**opts)

		sys.stderr.write(" --After run funcName:{},type:{}\n".format(funcName,type(dd)))
		sys.stderr.write(" --show_page OUTPUT:\n{}".format(dd)[:200]+"\n")

		#----------------------------------------------------------------

		# Use htmlName page is specified OR {pageBase}.html page
		if htmlName is None: 
			htmlName='{}.html'.format(pageBase)
		sys.stderr.write(" --htmlPage:{}\n".format(htmlName))

		# Use apitest.html as default page if htmlName page does not exist
		if 'templates/'+htmlName not in glob('templates/*html'):
			htmlName='apitest.html'

		# Process corresponding page
		ret = process_page(htmlName, dd, **opts)

		#----------------------------------------------------------------
	except Exception as e:
		sys.stderr.write("**ERROR: {} @ {}\n".format(str(e),"show_page"))
		#ret= str(e)
		ret= err_page(description=str(e)+"@ show_page")
	return ret

def process_page(htmlName, dd, **opts):
	# Processing output 
	from bson import json_util as json
	if isinstance(dd,dict):
		datax=json.dumps(dd,default=dtCvt)
		dd.update(datetime=datetime,pd=pd,s2dt=s2dt)
		ret= render_template(htmlName,datax=datax,**dd)
	elif isinstance(dd,list):
		opts.update(datetime=datetime,pd=pd,s2dt=s2dt)
		ret= render_template(htmlName,mtx=dd,**opts)
	else:
		opts.update(datetime=datetime,pd=pd,s2dt=s2dt)
		ret= render_template(htmlName, content=dd,**opts)
		sys.stderr.write(" --processing page:{}:opts:\n{}\n".format(htmlName,opts))

	if isinstance(dd,dict) and 'submit' in dd: 
		ret = make_response(ret)
		if dd['submit'] == 'login':
			ret.set_cookie('userID', dd['user'])
		elif dd['submit'] == 'logout':
			ret.delete_cookie('userID')
		elif dd['submit'] == 'delete':
			ret.delete_cookie('userID')


	return ret

def page_api(**optx):
	opts={"search":"comment","outTF":False}
	opts.update(optx)
	dfTF = opts.pop('dfTF',False)
	opts.update(debugTF=True)
	pqint("===page_api INPUT:\n",opts, file=sys.stderr)
	data=wrap_alanapi(**opts)
	if isinstance(data,pd.DataFrame) and dfTF is False:
		dd=data.to_json(orient='records',force_ascii=False)
	elif isinstance(data,dict):
		dd={x:dtCvt(y) for x,y in data.items()}
	elif isinstance(data,list):
		for j in range(len(data)):
			if not isinstance(data[j],dict):
				continue
			for x,y in data[j].items():
				data[j][x]=dtCvt(y)
		dd=data
	else:
		dd=data
	return dd

def page_login(**optx):
	if 'submit' not in optx:
		return optx
	pqint(" -- page_login INPUT:\n",optx, file=sys.stderr)
	submit=optx.pop('submit','login')
	url = 'http://login.beyondbond.com/'
	if submit in ['logout','lost-password','reset-password','delete','signup']:
		url = url+submit
	dd = dict(submit=submit)
	try:
		ret = rPost(url=url,data=optx)
		userInfo= json.loads(ret.decode('utf-8'))
		pqint("==Authentication OUT:\n",userInfo, file=sys.stderr)
		if isinstance(userInfo,dict):
			dd.update(userInfo)
	except Exception as e:
		pqint("==WARNINGS Login:{}\n".format(str(e)), file=sys.stderr)
	pqint(" -- page_login OUTPUT:\n",dd, file=sys.stderr)
	return dd

def page_signup(**optx):
	return page_login(**optx)

def page_index(**optx):
	dd={'search': 'comment', 'output': 'dict','topic': 'hourly','outTF':False}
	optx.update(dd)
	#ret=wrap_alanapi(**optx)
	ret= page_api(**optx)
	if len(ret)>0:
		optx.update(ret[0])
	return optx

def page_report(**optx):
	topic,subtopic,search=getKeyVal(optx,['topic','subtopic','search'],['report',None,'comment'])
	optx.update(search=search,topic=topic,subtopic=subtopic,output='json')
	pqint(" --page_report INPUT:\n",optx, file=sys.stderr)
	if subtopic =='company':
		optx.update(topic='news')
	ret= page_api(**optx)
	pqint(" --page_report OUTPUT:\n",type(ret),"{}".format(ret)[:30],file=sys.stderr)
	if isinstance(ret,list) and len(ret)>0 and subtopic != 'company':
		return ret[0]
	else:
		return ret

def page_performance(**optx):
	optx.update(search='quote',topic='detailed',ticker='AAPL')
	ret = page_api(**optx)
	pqint(" --page_report OUTPUT:\n",type(ret),"{}".format(ret)[:30],file=sys.stderr)
	if isinstance(ret,list) and len(ret)>0: 
		return ret[0]
	else:
		return ret

def page_news(**optx):
	topic,subtopic,search=getKeyVal(optx,['topic','subtopic','search'],['news',None,'comment'])
	optx.update(search=search,topic=topic,subtopic=subtopic)
	return page_api(**optx)

def page_portfolio_OLD(**optx):
	opts=dict (search='comment',output='html',outTF=True)
	opts.update(optx)
	return page_api(**opts)

def page_portfolio(**optx):
	opts=dict (search='comment',dfTF=True)
	opts.update(optx)
	dd= page_api(**opts)
	if isinstance(dd,pd.DataFrame):
		cfm = {'marketCap': lambda x: "{:,.2f}B".format(x/10**9),'ytdRtn': "{:,.2%}".format, 
			'W.A.Return': "{:,.2%}".format,'AvgReturn': "{:,.2%}".format}
		cfm.update(forwardPE="{:,.2f}".format)
		cfm.update(trailingPE="{:,.2f}".format)
		cfm.update(close="{:,.2f}".format)
		cfm.update(xclose="{:,.2f}".format)
		ret = dd.to_html(index=False,formatters=cfm)
	else:
		ret = ''
	#allow adding a comment:cmt here, TBD @ Sat Oct 24 21:23:48 EDT 2020
	cmt=''
	ret = cmt + ret
	return ret

def page_globalmacro(**optx):
	return page_api(**optx)

def page_apitest(**optx):
	return page_api(**optx)

def wrap_request(request):
	dd={}
	optGet = request.args.to_dict()
	pqint("===request GET Input:\n",optGet, file=sys.stderr)
	optPost = request.get_json()
	pqint("===request Post Input:\n",optPost,request.form, file=sys.stderr)
	user = request.cookies.get('userID')
	pqint("===request cookies user:{}".format(user) , file=sys.stderr)
	if optPost is not None:
		dd.update(optPost)
	if request.form is not None:
		dd.update(request.form.to_dict(flat=True))
	if optGet is not None:
		dd.update(optGet)
	if user is not None:
		dd.update(user=user)
	return dd

def wrap_alanapi(**opts):
	try:
		dd, _ = opt_alanapi()
		dd.update(opts)
		fdLst=dd.pop('field',None)
		if 'ticker' in dd:
			tkLst=dd.pop('ticker','').split(',')
		else:
			tkLst=[]
		sys.stderr.write(" --alanapi INPUT:\n{}\n".format(dd))
		data=run_alanapi(tkLst,fdLst,**dd)
	except Exception as e:
		data=dict(err=str(e))
		sys.stderr.write("**ERROR:{} at wrap_alan\n".format(str(e)))
	return data

def run_page_alanapi(request):
	dd = wrap_request(request)
	return wrap_alanapi(**dd)

def find_lsi2nlg_list(findDct={'username':'system'},fieldLst=['tmplname'],dbname='ara',tablename='lsi2nlg'):
	if not isinstance(fieldLst,list):
		fieldLst = list(fieldLst)
	fieldDct = {x:1 for x in fieldLst}
	xg,_,_ = find_mdb(dbname=dbname,tablename=tablename,jobj=findDct,field=fieldDct)
	if len(xg)<1:
		return []
	else:
		df=pd.DataFrame(xg)
		ky = fieldLst[0]
		return df[ky].sort_values().unique()

def find_lsi2nlg_info(jobj={},fieldLst=['username','category','tmplname','lang'],dbname='ara',tablename='lsi2nlg'):
	findDct=subDict(jobj,fieldLst)
	mobj,clientM,err_msg = find_mdb(jobj=findDct,dbname=dbname,tablename=tablename)
	if len(mobj)>0:
		return (mobj[0])
	else:
		return []

def write_lsi2nlg_info(jobj,zpk=['username','category','tmplname','lang'],dbname='ara',tablename='lsi2nlg'):
	mobj,clientM,err_msg = write2mdb(jobj,dbname=dbname,tablename=tablename,zpk=zpk)
	return mobj

def argstr_tmplrpt(tmplstr, dux, *_xargs_):
	"""
	Update list of [_xargs_] command scripts to dict [dux] 
	and then apply [dux] to jinja2 template [tmplstr] and produce [tmplrpt]
	return text output [tmplrpt]
	"""
	try:
		duy=combine_cmd2dict(*_xargs_) # TBD:,_xdu_=dux)
		if duy:
			dux.update(duy)
		#dux.update({'udfStr':udfStr,'roundUSD':roundUSD})
		tmplrpt = jj_fmt(tmplstr,dux)
	except Exception as e:
		tmplrpt = str(e)
	return tmplrpt

def argstr_set(gd,excl_ky=[]):
	"""
	assign dict [gd] keys as variables exclude [excl_ky] key list
	"""
	argstr=''
	for j,(ky,va) in enumerate(gd.iteritems()):
		if ky in excl_ky:
			continue
		if isinstance(va,StringTypes) and ky != "ts":
			argstr += "{}='{}';\n".format(ky,va)
		elif isinstance(va, (float, int, list, dict, tuple)):
			argstr += "{}={};\n".format(ky,va)
	return argstr

def run_j2ts(optGet={},optPost={}):
	pqint(optGet,optPost, file=sys.stderr)
	dd = {}
	if optPost is not None:
		dd.update(optPost)
	if optGet is not None:
		dd.update(optGet)
	if 'j2ts' in dd:
		j2ts = dd['j2ts'] 
		dd = subDict(dd,['j2ts'],reverseTF=True)
	else:
		j2ts = 'Usage of /?key1=value1&key2=value2 ...'
	return jj_fmt(j2ts,dd,j2onlyTF=True)

def tk_video(ticker, lang):
	xcmd = ('./stock_ticker2video.sh {} {}').format(ticker, lang)
	pqint('RUNNING ', xcmd, file=sys.stderr)
	p = Popen(xcmd, shell=True, bufsize=1024, stderr=PIPE, stdout=PIPE)
	try:
		outs, errs = p.communicate()
		return True
	except:
		p.kill()
		outs, errs = p.communicate(timeout=30)

	return False

def upd_video(name, lang):
	vdLst = sorted(iglob(('static/{}_{}.*.mp4').format(name, lang)), key=os.path.getmtime, reverse=True)
	# create the video if it does not exists 
	if len(vdLst) < 1:
		upd_flg = True
	else:
		fname = vdLst[0]
		xtime = datetime.datetime.fromtimestamp(float(fname.split('.')[-2:][0]))
		ctime = datetime.datetime.now()
		# re-create video for not today
		if ctime.strftime('%Y%m%d') != xtime.strftime('%Y%m%d'):
			upd_flg = True
		else: 
			# re-create video for older than 6-hour
			if (ctime - xtime).seconds / 3600 >= 12:
				upd_flg = True
			else:
				upd_flg = False
	if upd_flg is True:
		TF = tk_video(name, lang)
		vdLst = sorted(iglob(("static/{}_{}.*.mp4").format(name, lang)), key=os.path.getmtime, reverse=True)
		if len(vdLst) < 1:
			pqint("**ERROR: {} @ {}".format(fname,'stock_ticker2video','upd_video()'), file=sys.stderr)
			return ''
		fname = vdLst[0]
		for xf in vdLst[2:]:
			os.remove(xf)

	else:
		pqint(("\t{} exist @ {}").format(fname, xtime), file=sys.stderr)
	return os.path.basename(fname)


def qt_result(name,lang,videoYN='0',errMsg=''):
	from yh_chart import yh_hist_query as yhq
	#from dateutil.parser import parse as Timestamp
	name=str(name.upper())
	videoYN=str(videoYN)
	d = dict(name=name,lang=lang,videoYN=videoYN,len=len,enumerate=enumerate,error_message=errMsg)
	if videoYN == '1':
		pqint("\tvideoYN:{} @ {}".format(videoYN, 'qt_result'), file=sys.stderr)
		try:
			e=''
			fname = upd_video(name, lang)
		except Exception as e:
			fname=''
		if len(fname)<1:
			errMsg = "**ERROR: No [{}_{}] video {} @ {}".format(name,lang,str(e),'upd_video')
		d.update(error_message=errMsg,mp4path=fname)
		return d
	pqint("\tvideoYN:{} @ {}".format(videoYN, 'qt_result'), file=sys.stderr)
	try:
		qt = yhq([name],types='quote',rawTF=True,dbname=None)[0]
		pqint("===QT:\n{}".format(qt), file=sys.stderr)
	except Exception as e:
		pqint("===yh_hist_query: {}".format(str(e)), file=sys.stderr)
		qt = {}
	if 'regularMarketTime' in qt:
		qt['lastUpdate'] = datetime.datetime.fromtimestamp(qt['regularMarketTime']).strftime("%Y/%m/%d %H:%M:%S")
	d.update(qt)
	try:
		d['quote'] = render_template('tp_quote.html',quote=qt)
	except Exception as e:
		pqint("===render_template: {}".format(str(e)), file=sys.stderr)
	if 'longName' in qt:
		d['company'] = qt['longName']
	if 'epsTrailingTwelveMonths' in qt:
		d['earnings'] = qt['epsTrailingTwelveMonths']
	if 'trailingPE' in qt:
		d['financials'] = qt['trailingPE']
	return d

def run_rmc(jobj):
	"""
	run Remote-Market-Commentary based on input dict: [jobj]
	with keys: tmplstr,argstr,ticker,tmplrpt,username,category,tmplname,action
	return string for Market Commentary
	"""
	global username, ticker, lang , dirname
	if 'username' in jobj:
		username=jobj['username']
	if 'ticker' in jobj:
		ticker=jobj['ticker']
	if 'lang' in jobj:
		lang=jobj['lang']
	dirname='templates/'
	jobj.update(dirname=dirname)
	jobj.update(mp3YN=False)
	jobj.update(start=dt2ymd(next_date()))
	# run rmc based on tmplstr
	if str(jobj['tmpltype']) == 'pppscf' and str(jobj['action']) == 'run':
		jobj['tmplrpt'] = batch_pppscf([jobj['ticker']],optx={"j2ts":jobj['tmplstr'],"lang":jobj['lang'],"debug":True,"saveDB":False,"category":jobj['category']})
		# run_alan_pppscf.batch_pppscf() 1st then import gData in batch_pppscf()
		from _alan_pppscf import gData
		jobj["retcode"]="run successfully"
	elif str(jobj['tmpltype']) == 'fcs' and str(jobj['action']) == 'run':
		jobj['tmplrpt'] = batch_ohlc_fcs([jobj['ticker']],optx={"j2ts":jobj['tmplstr'],"lang":jobj['lang'],"debug":True,"saveDB":False,"category":jobj['category']})
		from _alan_ohlc_fcs import gData
		jobj["retcode"]="run successfully"
	elif str(jobj['tmpltype']) == 'eps' and str(jobj['action']) == 'run':
		from fcs_eps_price import run_eps_fcs
		gData = run_eps_fcs([jobj['ticker']])[0]
		jobj['tmplrpt'] = argstr_tmplrpt(jobj['tmplstr'],gData,jobj['argstr'].replace("\n",''))
		jobj["retcode"]="run successfully"
	elif str(jobj['tmpltype']) == 'mkt' and str(jobj['action']) == 'run':
		# Same as command line:
		# headline_writer.py --lang=cn --extra_xs='onTheFly=True;j2ts="{% include \"daily_briefing_cn.j2\" %}";dirname="templates"' --start=20190705 2>/dev/null

		from headline_writer import generate_headline
		optx=dict(j2ts=jobj['tmplstr'],onTheFly=True)
		optx.update(jobj)
		if jobj['argsflg']=='1' and len(jobj['argstr'])>0:
			gdd={}
			exec(jobj['argstr'],globals(),gdd)
			optx.update(gdd)
		try:
			ret = generate_headline(**optx)
		except Exception as e:
			err_msg = "**ERROR:{} @ {} ".format(str(e),"generate_headline")
			pqint(err_msg, file=sys.stderr)
			ret=err_msg
		jobj['tmplrpt'] = ret
	elif str(jobj['tmpltype']) == 'test' and str(jobj['action']) == 'run':
		optx={'j2ts_header':jobj['tmplstr']}
		optx.update(jobj) 
		try:
			# use 'jobj' as internal object and update jobj's key/val into **optx 
			pqint("===username: {}, argflg: {},argstr: {}".format(username,jobj['argsflg'],jobj['argstr']),file=sys.stderr)
			if jobj['argsflg']=='1' and len(jobj['argstr'])>0:
				gdd={}
				xstr = jobj['argstr'] 
				exec(xstr,globals(),gdd)
				if 'jobj' in gdd:
					optx.update(gdd['jobj'])
					pqint("Add additional:{}".format(gdd['jobj']),file=sys.stderr)
				else:
					optx.update(gdd)
					pqint(gdd,file=sys.stderr)
		except Exception as e:
			pqint("**ERROR:{} @ {} ".format(str(e),"exec argstr"), file=sys.stderr)
		pqint(optx)
		ret = jj_fmt(jobj['tmplstr'],**optx)
		jobj['tmplrpt'] = ret
	# TBD: testing 'get' instead of lsi2nlg_calc()
	elif str(jobj['action']) == 'get':
		mobj = find_lsi2nlg_info(jobj=jobj)
		jobj.update(mobj)
		jobj['argsflg']=1
	# TBD: testing 'save' instead of lsi2nlg_calc()
	#elif str(jobj['action']) == 'save':
	#	mobj = write_lsi2nlg_info(jobj)
	#	jobj['argsflg']=1
	else:
		jobj['argstr'] = jobj['argstr'].replace("\n",'')
		jobj = lsi2nlg_calc(jobj)
		jobj['argsflg']=1
	# run rmc with additional argstr command
	if str(jobj['tmpltype']) in ['eps','fcs','pppscf'] and str(jobj['action']) == 'run':
		if 'itnstr' in jobj:
			try:
				jobj['itnstr'] = argstr_set(gData,['fp'])
			except Exception as e:
				pqint("**ERROR:{} @ {} ".format(str(e),"Assign Internal Variables"), file=sys.stderr)
		try:
			if 'argsflg' in jobj and int(jobj['argsflg'])==1:	
				jobj['tmplrpt'] = argstr_tmplrpt(jobj['tmplstr'],gData,jobj['itnstr'],jobj['argstr'].replace("\n",''))
		except Exception as e:
			pqint("**ERROR:{} @ {} ".format(str(e),"Apply Additional Variables"), file=sys.stderr)
			jobj["retcode"]=str(e)
	jobj["lastmod"]=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	jstr=json.dumps(jobj)
	return jstr

def topic_theme_majorplayer(**opts):
	'''
	use 'themename' as tablename
	'''
	from _alan_str import find_mdb
	fund=getKeyVal(opts,'fund','renaissance-technologies-llc')
	clientM=getKeyVal(opts,'clientM',None)
	dbname=getKeyVal(opts,'dbname','ara')
	themename=getKeyVal(opts,'themename','topic_theme_majorplayer')
	dfTF=getKeyVal(opts,'dfTF',True)
	jobj = dict(fund=fund)
	data0,mDB,errmsg = find_mdb(jobj,tablename=themename,dbname=dbname,sortLst={"funddate"},limit=1,dfTF=False)
	mxdate=data0[0]['funddate']
	jobj.update({"funddate":mxdate})
	df,mDB,errmsg=find_mdb(jobj,tablename=themename,dbname=dbname,dfTF=dfTF)
	colX = ['ticker','close','pbdate','marketCap',
		'changePercent', 'CurrentShares','percentPos','SharesChangePercent',
		'fiftyTwoWeekRange','shortName',
		'fund','funddate','pbdt']
	colDc = {'close':'closePrice','pbdate':'closeDate','shortName':'company',
		'SharesChangePercent':'SharesChg%','changePercent':'Chg%',
		'percentPos':'Position%'
		}
	df=subDict(df,colX)
	df=renameDict(df,colDc)
	df = df.dropna(subset=['closeDate'])
	df['closeDate'] = [int(x) if not np.isnan(x) else 0 for x in df['closeDate'].values]
	return df

def topic_theme_media(**opts):
	from _alan_str import find_mdb
	jobj = {"buyCount":{"$gt":5}}
	dtmp,mDB,errmsg=find_mdb(jobj,tablename='topic_theme_media',dbname='ara',dfTF=True)
	df = dtmp.sort_values(by='buyCount',ascending=False)
	colX = ['ticker','buyDate','marketCap','buyPrice','closePrice','buyChg%','buyCount','dayChg%','EPS','pbdate','Range52Week','Company','sector','industry','pubDate']
	if 'marketCap' in df:
		df['marketCap'] = df['marketCap'].astype(float)
	df=subDF(df,colX)
	return df

def topic_theme_media_OLD(**opts):
	from _alan_str import find_mdb
	subtab=getKeyVal(opts,'subtab','blacklist')
	if subtab.lower() == 'whitelist':
		jobj = {"rrt":{"$gt":10}}
		dtmp,mDB,errmsg=find_mdb(jobj,tablename='topic_theme_media',dbname='ara',dfTF=True)
		df = dtmp.sort_values(by='rrt',ascending=False).iloc[:100]
	elif subtab.lower() == 'conflict':
		dtmp,mDB,errmsg=find_mdb({},tablename='topic_theme_media',dbname='ara',dfTF=True)
		a1=list(dtmp.query("rrt>=1.0")['ticker'].unique())
		a2=list(dtmp.query("rrt<=-1.0")['ticker'].unique())
		aLst=set(a1).intersection(set(a2))
		df = dtmp.loc[dtmp['ticker'].isin(aLst)].sort_values(by='ticker')
	else:
		jobj = {"rrt":{"$lt":-10}}
		dtmp,mDB,errmsg=find_mdb(jobj,tablename='topic_theme_media',dbname='ara',dfTF=True)
		df = dtmp.sort_values(by='pbdate',ascending=False).iloc[:100]
		colX=["ticker","pbdate","start","end","sPrice","ePrice","rrt","Price","SegmentDscript","CallDscript","Company"]
		df=subDF(df,colX)
	return df

def topic_theme_ipo(**opts):
	from _alan_str import find_mdb
	subtab=getKeyVal(opts,'subtab','')
	opts.pop('output',None) # not output type assigned
	updTF=opts.pop('updTF',False)
	jobj=json.loads(opts.pop('jobj','{}'))
	sector=opts.pop('sector','')
	if len(sector)>5:
		sLst=sector.split(',')
		jobj.update(sector={"$in":sLst})

	#df = get_ipoReport(updTF=updTF,**opts) # DEPRECATED
	#df = create_topic_theme_ipo(updTF=updTF,**opts) # SETUP in crontab

	tablename=opts.pop('tablename',None)
	if tablename is None:
		tablename='topic_theme_ipo'
	df,_,_ = find_mdb(jobj,dbname='ara',tablename=tablename,dfTF=True)
	colX = ['ticker','ipoDate','marketCap','ipoPrice','Price','closePrice','ipoChg%','dayChg%','EPS','Range52Week','Company','sector','industry','pubDate']
	df=subDict(df,colX)

	#-onTheFly run, not used
	#-run daily crontab to create 'topic_theme_ipo' table via  
	#-python3 -c "from ipoReport import create_topic_theme_ipo as ctt;ctt()"

	return df

def run_topic_theme(dfTF=True,**opts):
	subtopic=getKeyVal(opts,'subtopic','majorplayer')
	funcName = "_".join(["topic","theme",subtopic]) 
	funcArg = globals()[funcName] if funcName in globals() else topic_theme_majorplayer
	try:
		pqint(" --dfTF:{},{},funcArg:{} @{}".format(dfTF,funcArg,opts,'run_topic_theme()'),file=sys.stderr)
		df = funcArg(**opts)
		if dfTF:
			return df
		cfm = {'marketCap': "{:,.0f}".format}
		ret = df.to_html(index=False,formatters=cfm)
		sty = "<style>.dataframe {text-align:right;}</style>\n"
		return sty+ret
	except Exception as e:
		pqint(str(e))
		return str(e)

def run_api(jobj,engine=None):
	pd.options.display.float_format = '{:,.2f}'.format
	sty = "<style>.dataframe {text-align:right;}</style>\n"
	ret="""Usage of:
	<H3>
	?topic='TOPIC'
	</H3>
	Where
	<PRE>
	TOPIC = [ipo|theme|priceRange|top10|utdEarningsList|mongo_search|file|write2file|test] """
	topic=getKeyVal(jobj,'topic','').lower()
	if topic is None:
		return ret

	if topic == 'theme':
		subtopic=getKeyVal(jobj,'subtopic','majorplayer')

	if topic == 'theme' and subtopic in ['majorplayer','media','ipo']:
		jobj.update(subtopic=subtopic)
		return run_topic_theme(dfTF=False,**jobj)
	elif topic == 'pricerange':
		xqTmp= '''select * from (select ticker,close as price,"trailingPE" as "peRatio","marketCap"::float/1000000 as "marketCapMM","changePercent" as "change%%",change from yh_quote_curr where close>={} and close<={}) as a, (select ticker,company_cn,company,sector_cn from mapping_ticker_cik where act_code>0) as b where a.ticker=b.ticker order by price'''
		try:
			vr=jobj['range'].split(",") if 'range' in jobj else [60,70]
			if len(vr)>=2:
				vr=np.array(vr,dtype=float)
			lb,ub = (vr.min(),vr.max())
			xqr = xqTmp.format(lb,ub)
			df = sqlQuery(xqr)
			pd.options.display.float_format = '{:,.2f}'.format
			cfm = {'marketCapMM': "{:,.0f}".format}
			ret = df.to_html(formatters=cfm)
			return sty+ret
		except Exception as e:
			pqint(str(e))
			return str(e)
	elif topic == 'top10':
		return run_top10(jobj,engine=None)
	elif topic in ['utd_earnings_list','utdearningslist']:
		from utd_earnings_list import utd_earnings_list
		dd=dict(sector='Technology',pbdate='20190101')
		dd.update(subDict(jobj,['sector','pbdate']))
		df=utd_earnings_list(**dd).sort_values(by=['pbdate','marketCap'],ascending=[False,False])
		df.rename(columns={'quarter':'epochs'},inplace=True)
		cfm = {'marketCap': "{:,.0f}".format,'recdate':"{:.0f}".format,'epochs':"{:.0f}".format }
		ret = df.to_html(formatters=cfm)
		return sty+ret
	elif topic in ['daily_med','dailymed']:
		from dailyMed_api import drug2labelInfo as dli
		try:
			dd =  dli(jobj['drug_name'])
		except Exception as e:
			sys.stderr.write("**ERROR:{}\n".format(str(e)))
			return(str(e))
		#return dd
		ts="""{{drug_name}}
		<P>
		{{ sec_cn|join('</P><P>\n') }}
		</P>
		"""
		return jj_fmt(ts,**dd)
	elif topic == 'mongo_search':
		d=dict(dbname='ara',tablename='lsi2nlg',username='system',field='tmplname',ticker='AAPL')
		if len(jobj)>0:
			d.update(jobj)
		findDct={'username':d.get('username')}
		fieldLst = d.get('field').split(',')
		fieldDct = {x:1 for x in fieldLst}
		xg,_,_ = find_mdb(dbname=d['dbname'],tablename=d['tablename'],jobj=findDct,field=fieldDct)
		return xg
	elif topic == 'file':
		try:
			dirname = jobj['dirname'] if 'dirname' in jobj else "/apps/fafa/pyx/tst"
			if 'j2name' in jobj:
				fname= "{}/{}".format(dirname,jobj['j2name'])
				ret = open(fname).read()
			elif 'image' in jobj:
				fname= "{}/{}".format(dirname,jobj['image'])
				ret = open(fname).read()
		except Exception as e:
			sys.stderr.write("**ERROR:{}".format(str(e)))
	elif topic == 'write2file':
		# save 'j2name' to 'dirName'  
		pqint(jobj)
		try:
			dirName = "/apps/fafa/pyx/flask/rmc/templates"
			if 'j2name' in jobj and 'j2ts' in jobj:
				if 'dirname' in jobj:
					dirName = jobj['dirname']
				fname = "{}/{}".format(dirName,jobj['j2name'])
				fp = open(fname,'w+')
				fp.write(jobj['j2ts'])
				fp.close()
				pqint("===Save {} To {}".format(jobj['j2ts'],fname))
				ret = "Successfully save to {}".format(fname)
		except Exception as e:
			sys.stderr.write("**ERROR:{}".format(str(e)))
			ret = str(e)
	elif topic == 'test':
		if 'tmplrpt' in jobj:
			ret= jobj['tmplrpt']
	return ret

def run_top10(jobj,engine=None):
	xobj={}
	if jobj['category'] == 'backtest':
		xql = "SELECT * FROM ara_ranking_backtest_cn"
		df = sqlQuery(xql,engine=engine)
		xobj = df.to_dict(orient='records')
	return xobj
