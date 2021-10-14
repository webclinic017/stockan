#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage of:
python3 _alan_mp4.py daily_briefing --extra_xs="txtOnly=True"
python3 _alan_mp4.py intraday_briefing --extra_xs="txtOnly=True"
OR
python3 _alan_mp4.py daily_single_stock MDB --extra_xs="txtOnly=True;deltaTolerance=43200"
OR
python3 _alan_mp4.py daily_single_stock MDB 1

Description: mp4 factory
Version: 0.9
Last mod., Thu Oct 29 13:19:52 EDT 2020
"""
import sys,os
import re
from optparse import OptionParser
from glob import glob
import numpy as np
import pandas as pd
from _alan_calc import getKeyVal,subDict,saferun
from _alan_str import jj_fmt,upsert_mdb,find_mdb
from srt_convert import generateSubtitle
from subprocess import Popen,PIPE
import datetime
from pandas import Timestamp
if sys.version_info.major == 2:
	reload(sys)
	sys.setdefaultencoding('utf8')

def str2audio(s='',fname='',lang='cn',tempo=1.5,debugTF=False):
	'''
	create a audio file named 'fname' based on string 's'
	'''
	if any([len(s)<1,len(fname)<1]) is True:
		return '','**ERROR: No string for audio'
	glang='zh-CN' if lang=='cn' else lang
	xcmd="/usr/local/bin/gtts-cli -l {} -f - | sox -G -t mp3 - {} tempo {}".format(glang,fname,tempo)
	if debugTF:
		sys.stderr.write("SHELL_CMD:\n{} | {}\n".format(s,xcmd))
	p = Popen(xcmd,shell=True,bufsize=1024,stdout=PIPE,stdin=PIPE,stderr=PIPE)
	o, e = p.communicate(s.encode('utf-8'))
	if debugTF:
		sys.stderr.write("===SHELL_CMD:\n{}\n---OUT:\n{}\n---ERR:\n{}\n".format(xcmd,o,e))
	return o,e

def run_concat(listName='',videoLst=[],videoPath='',outdir='',lang='cn',debugTF=False):
	'''
	concatenate 'videoLst' mp4 files and save to 'videoPath'
	'''

	if len(outdir)<1:
		outdir=os.path.dirname(videoLst[0])
	vlst=[os.path.basename(x) for x in videoLst]
	listStr="file {} \n".format("\nfile ".join(vlst))
	mp4Name=os.path.basename(videoPath)
	if len(listName)<1:
		listName=re.sub('(_ALL)?(.mp4)?','',mp4Name)+'.list'
		listPath="{}/{}".format(outdir,listName)
	fp=open(listPath,"w")
	fp.write(listStr)
	fp.close()
	tempName=re.sub('(_ALL)?(.mp4)?','',mp4Name)+'_temp.mp4'
	sampleRate=24000 if lang =='en' else 22050 
	ffm ='''cd {outdir};'''
	ffm+='''ffmpeg -nostdin -f concat -i "{listName}" -c copy  -y "{tempName}";'''
	ffm+='''ffmpeg -nostdin -i {tempName} -c:v libx264 -pix_fmt yuv420p -flags +global_header -strict -2 -acodec aac -ar {sampleRate} -ab 32k "{mp4Name}" -y'''
	# need listName, tempName, mp4Name 
	xcmd=ffm.format(**locals())
	#return ffm
	try:
		p = Popen(xcmd,shell=True,bufsize=1024,stdout=PIPE,stdin=PIPE,stderr=PIPE)
		o, e = p.communicate()
		#This makes the wait possible
		p_status = p.wait()
	except Exception as e:
		sys.stderr.write("**ERROR: {} @ {}\n".format(str(e),'run_ffmpeg') )
		o=''
	if debugTF:
		sys.stderr.write("===SHELL_CMD:\n{}\n---OUT:\n{}\n---ERR:\n{}\n".format(xcmd,o,e))
	return o,e

def run_ffmpeg(chartPath='',audioPath='',videoPath='',srtPath='',lang='cn',debugTF=False):
	'''
	Create  mp4 file 'videoPath'
	  based on chart,audio and subtitle files w.r.t.
	  'chartPath', 'audioPath', 'srtPath'
	'''
	if lang=='en' :
		fs=22
		fb="blue"
		fc="white"
		spd=3.5
		sampleRate=24000
	else :
		fs=24
		fb="red"
		fc="white"
		spd=5.5
		sampleRate=22050

	#ffm='''ffmpeg -nostdin -loop 1 -i "{chartPath}" -i "{audioPath}" -shortest -tune stillimage -c:v libx264 -pix_fmt yuv420p -flags +global_header -c:a copy -movflags +faststart -vf "drawbox=y=ih-h-0:h=72:color={fb}@0.85:t=fill","subtitles={srtPath}:force_style='fontsize={fs},MarginV=5'" -y "{videoPath}"'''
	ffm='''ffmpeg -nostdin -loop 1 -i "{chartPath}" -i "{audioPath}" -shortest -tune stillimage -c:v libx264 -pix_fmt yuv420p -flags +global_header -c:a copy -movflags +faststart -vf "subtitles={srtPath}:force_style='fontsize={fs},MarginV=5'" -y "{videoPath}"'''
	xcmd=ffm.format(**locals())
	#return ffm
	try:
		p = Popen(xcmd,shell=True,bufsize=1024,stdout=PIPE,stdin=PIPE,stderr=PIPE)
		o, e = p.communicate()
	except Exception as e:
		sys.stderr.write("**ERROR: {} @ {}\n".format(str(e),'run_ffmpeg') )
		o=''
	if debugTF:
		sys.stderr.write("===SHELL_CMD:{}\n---OUT:{}\n---ERR:{}\n".format(xcmd,o,e))
	return o,e

def dat2rptTxt(ts='',dd={},lang='cn',mp3YN=False,ctrlSTRUCT=None,debugTF=False,_pn_={},**optx):
	rptTxt=''
	try:
		if ctrlSTRUCT is not None and ctrlSTRUCT in ['include','extends']:
			ts='{{% {} "{}" %}}'.format(ctrlSTRUCT,ts)
		rptTxt = jj_fmt(ts,dd,lang=lang,mp3YN=mp3YN,_pn_=_pn_,**optx)
		sys.stderr.write("=====dat2rptTxt: {}\n{}\n".format(rptTxt,optx) )
	except Exception as e:
		sys.stderr.write("**ERROR: {} @ {}\n".format(str(e),'dat2rptTxt') )
	return rptTxt

def dat2audioTxt(ts='',dd={},lang='cn',mp3YN=False,debugTF=False,_pn_={},**optx):
	return dat2rptTxt(ts=ts,dd=dd,lang=lang,mp3YN=mp3YN,debugTF=debugTF,_pn_=_pn_,**optx)

def dat2chart(datax={},title='',outdir='.',debugTF=False,**optx):
	chartPath="{}/{}.svg".format(outdir,title)
	return chartPath

def txt2audio(mp3comment='',audioPath='',lang='cn',tempo=1.5,debugTF=False,**optx):
	o,e = str2audio(mp3comment,fname=audioPath,lang=lang,tempo=tempo,debugTF=debugTF)
	return audioPath

def audio2srt(comment='',audioPath='',srtPath='',strTF=True,debugTF=False,lang='cn',**optx):
	ret=generateSubtitle(textFile=comment,audioFile=audioPath,srtFile=srtPath,lang=lang,strTF=strTF)
	return ret

def audio2video(audioPath='',videoPath='',srtPath='',chartPath='',datax={},lang='cn',debugTF=False,**optx):
	o,e = run_ffmpeg(audioPath=audioPath,videoPath=videoPath,srtPath=srtPath,chartPath=chartPath,lang=lang,debugTF=debugTF)
	return videoPath

def mp4_tst(ts='',category='EoD',title='daily_briefing',datax={},debugTF=False,**optx):
	ret = tmpl2mp4(ts=ts,category=category,title=title,debugTF=debugTF,**optx)
	return ret

def set_ht(_pn_):
	fx = lambda m,y: [m[x][y] for x in m['tmplLst'] if y in m[x]]
	ht = fx(_pn_,'ht3')
	if len(ht)<1:
		ht = fx(_pn_,'ht2')
	if len(ht)<1:
		ht = fx(_pn_,'ht1')
	headTitle='' if len(ht)<1 else ht[0]
	_pn_.update(headTitle=headTitle)
	return _pn_
	
def saveBriefing(_pn_,tablename="mkt_briefing_details",zpk=['category','templateName','rpt_time','lang']):
	fx = lambda m,y: [m[x][y] for x in m['tmplLst'] if y in m[x] and m[x][y] is not None]
	try:
		if 'headTitle' not in _pn_:
			_pn_ = set_ht(_pn_)
		if 'comment' not in _pn_:
			ret = "\n".join(fx(_pn_,'comment'))
			_pn_.update(comment=ret)
		if 'mp3comment' not in _pn_:
			m3ret = "\n".join(fx(_pn_,'mp3comment'))
			_pn_.update(m3comment=m3ret)
		ret,_,emsg = upsert_mdb([_pn_],zpk=zpk,dbname='ara',tablename=tablename,ordered=False,jsonObjReloadTF=False)
		sys.stderr.write("==errmsg:SAVE TO {}:{} {}\n".format(tablename,_pn_['pbdt'],str(emsg)))
	except Exception as e:
		sys.stderr.write("**ERROR:INSERT INTO {}: {} of _pn_:\n{}\n".format(tablename,str(e),_pn_))
	return _pn_

@saferun
def tmpl2mp4(ts='',category='ItD',title='intraday_briefing',datax={},debugTF=False,_pn_={},**optx):
	'''
	Create comment, mp3comment based on 'title' and svg and 
	save them into into MDB table: 'mkt_briefing_details' (for 'intraday_briefing' and 'daily_briefing')
	or 'daily_single_stock' (for title='daily_single_stock') 
	'''

	if title[-9:]=='_briefing':
		tablename="mkt_briefing_details"
		rptTxt = dat2rptTxt(ts=ts,_pn_=_pn_,debugTF=debugTF,**optx)
		zpk = ['category','templateName','rpt_time','lang']
		sys.stderr.write("==SAVE {} to MDB table:{}\n".format(title,tablename))
	elif title=='daily_single_stock':
		dbname='ara'
		tablename=title
		deltaTolerance= optx.pop('deltaTolerance',43200)
		ticker=getKeyVal(optx,'ticker','')
		o,m,e=find_mdb(dict(ticker=ticker),dbname=dbname,tablename=tablename,sortLst=['pbdt'])
		if len(o)>0 and 'pbdt' in o[0]:
			pbdtMod=o[0]['pbdt']
			deltaPassed=int(pd.Timedelta(pd.datetime.now()-pbdtMod).total_seconds())
			if deltaPassed<=deltaTolerance:
				sys.stderr.write(" --pbdtMod:{},deltaPassed:{},deltaTolerance:{}\n".format(pbdtMod,deltaPassed,deltaTolerance))
				sys.stderr.write(" --Directly use comment from table: {}\n".format(tablename))
				return o[0]
		rptTxt = dat2rptTxt(ts=ts,_pn_=_pn_,debugTF=debugTF,**optx)
		_pn_['headTitle']=_pn_['stock_performance']['comment'];
		ticker=_pn_['ticker']=_pn_['stock_performance']['ticker'];
		_pn_['category']=category
		_pn_['templateName']=title
		_pn_['pbdt']=_pn_['stock_performance']['pbdt'];
		_pn_['lang']=getKeyVal(optx,'lang','cn')
		zpk=['ticker']
		sys.stderr.write("==SAVE {}:{} to MDB table:{}\n".format(title,ticker,tablename))
	else:
		return {}
	_pn_.update(title=title)
	saveBriefing(_pn_,tablename=tablename,zpk=zpk)
	if debugTF:
		sys.stderr.write("=====ts:\n{}\n".format(ts))
		sys.stderr.write("=====optx:\n{}\n".format(optx))
		sys.stderr.write("=====rptTxt:\n{}\n".format(rptTxt))
		sys.stderr.write("=====_pn_:\n{}\n".format(_pn_))
		sys.stderr.write("=====keys:\n{}\n".format(_pn_.keys()))
	if 'tmplLst' in _pn_:
		tmplLst = _pn_['tmplLst']
		sys.stderr.write("=====tmplLst: {}\n".format(tmplLst))
		for tname in tmplLst:
			dx = _pn_[tname]
			sys.stderr.write("===Chartpath: {}:{}\n".format(tname,dx['chartpath']))
	else:
		sys.stderr.write("*ERROR:tmplLst not found in {}\n".format(_pn_.keys()))
		return {}
	txtOnly=optx.pop('txtOnly',False)
	if txtOnly is True:
		sys.stderr.write("=====RUN Text Only: {}\n".format(txtOnly))
		return _pn_
	sys.stderr.write("=====RUN pn2mp4: {}\n".format(tmplLst))
	dpn = pn2mp4(_pn_=_pn_,zpk=zpk,**optx)
	return dpn

def pn2mp4(_pn_={},zpk=[],debugTF=False,**optx):
	'''
	convert _pn_ that contail svg, comment, mp3comment into relevant mp3 and mp4 files
	and save file locations and save to MDB table: 'mkt_briefing_media' or 'daily_single_stock_media' 
	'''
	if 'tmplLst' not in _pn_:
		return {}
	tmplLst = _pn_['tmplLst']
	title = _pn_['title']
	dpn=subDict(_pn_,zpk+['title','pbdt','headTitle','intraday_headline','daily_headline','comment','mp3comment'])
	videoLst=[]
	j=0
	for tname in tmplLst:
		if tname not in _pn_:
			sys.stderr.write("===NotFound: @ {}\n".format(tname))
			continue
		sys.stderr.write("===Running: {}\n{}\n".format(tname,_pn_[tname]))
		try:
			dx = _pn_[tname]
			if 'chartpath' not in dx or dx['chartpath'] is None:
				continue
			chartpath= dx['chartpath']
			xtmp=re.sub("(.svg)?","",chartpath)
			outdir,mpname=os.path.dirname(xtmp),os.path.basename(xtmp)
			if len(mpname)<1:
				sys.stderr.write("**WARNING: @ {}\t{}\n".format(tname,"Filename not exists"))
				continue
			dx.pop('title',None)
			vk = [k for k in dx]
			for k in vk:
				v = dx[k]
				if isinstance(v,(dict,tuple,list,pd.DataFrame)):
					dx.pop(k,None)
					#dx.update(k=None)
			txtChart2audioSrtVideo(tmplname=tname,mpname=mpname,tmplobj=dx,debugTF=debugTF,**optx)
			dpn.update({tname:dx})
			videoLst.append(dx['videoPath'])
			sys.stderr.write("\n==={}:{} successfully created!\n".format(tname,dx['videoPath']))
		except Exception as e:
			sys.stderr.write("**ERROR: @ {}\n\t{}\n".format(tname,str(e)))
			continue
		j=j+1
	if j<1:
		return {}
	sys.stderr.write("===Total:{}: {}\n".format(j,videoLst))
	try:
		xpoch = mpname.split("_")[-1]
		if title=='daily_single_stock':
			ticker = getKeyVal(_pn_,'ticker',None)
			videoPath='{}_{}_{}_ALL.mp4'.format(title,ticker,xpoch)
		else:
			videoPath='{}_{}_ALL.mp4'.format(title,xpoch)
		run_concat(videoLst=videoLst,outdir=outdir,videoPath=videoPath)
		#dpn.update(comment=rptTxt)
		dpn.update(videoPath=videoPath)
		sys.stderr.write("===Combine: {} To {}\n".format(videoLst,videoPath))
		if len(glob(outdir+"/"+videoPath))>0:
			sys.stderr.write("===videoLst: {}\n".format(videoLst))
			sys.stderr.write("===videopath: {} successfully created!\n".format(videoPath))
		else:
			sys.stderr.write("**ERROR: {} not created!\n".format(videoPath))

		# save to MDB 'dbname'::'tablename'
		dbname=optx.pop('dbname','ara')
		tablename=optx.pop('tablename','')
		if title[-9:]=='_briefing':
			tablename='mkt_briefing_media'
		elif title=='daily_single_stock':
			tablename=title+'_media'
		if len(tablename)>5:
			ret,_,emsg = upsert_mdb([dpn],zpk=zpk,dbname=dbname,tablename=tablename,ordered=False)
			sys.stderr.write("==errmsg:SAVE TO {} @ {}, STATUS:{}\n".format(tablename,dpn['pbdt'],str(emsg)))
	
		sys.stderr.write("==SUCCESS videoLst:{}, mp4:{}\n".format(videoLst,mpname))
	except Exception as e:
		sys.stderr.write("**ERROR: @ {}:{}\n\t{}\n".format(videoLst,mpname,str(e)))
	return dpn

def txtChart2audioSrtVideo(tmplname='EoD',mpname='',tmplobj={},debugTF=False,**optx):
	'''
	Create srt, mp3 & mp4 given 'rptTxt', 'audioTxt' and 'chartpath'
	and save them in 'srtPath','audioPath' and 'videoPath' respectively
	'''
	vTF=[x in tmplobj for x in ['chartpath','comment','mp3comment']]
	if any(vTF) is False:
		return {}
	outdir=getKeyVal(optx,'outdir','.')
	audioPath="{}/{}.mp3".format(outdir,mpname)
	videoPath="{}/{}.mp4".format(outdir,mpname)
	srtPath="{}/{}.srt".format(outdir,mpname)
	rptTxt = tmplobj['comment']
	chartPath = tmplobj['chartpath']
	audioTxt = tmplobj['mp3comment']
	audioPath = txt2audio(mp3comment=audioTxt,audioPath=audioPath,debugTF=debugTF,**optx)
	srtTxt = audio2srt(comment=rptTxt,audioPath=audioPath,srtPath=srtPath,strTF=True,debugTF=debugTF,**optx)
	videoPath = audio2video(videoPath=videoPath,audioPath=audioPath,srtPath=srtPath,chartPath=chartPath,debugTF=debugTF,**optx)
	tmplobj.update(tmplname=tmplname,title=mpname,rptTxt=rptTxt,audioTxt=audioTxt,audioPath=audioPath,srtPath=srtPath,chartPath=chartPath,videoPath=videoPath)
	return tmplobj

def find_file(fn):
	if isinstance(fn,list):
		for f in fn:
			ret = glob(f)
			if len(ret)>0:
				return ret
		return []
	else:
		return glob(fn)
	
def tmpl_wrapper(tmplname,debugTF=False,_pn_={},tmplLst=[],lang='cn',ctrlSTRUCT='include',outdir='US/mp3_hourly',dirname='templates',**optx):
	fn="{dirname}/{tmplname}_cn.j2\n{dirname}/{tmplname}.j2".format(dirname=dirname,tmplname=tmplname).split("\n")
	tLst = find_file(fn)
	if len(tLst)<1:
		return {}
	ts = os.path.basename(tLst[0])
	dpn = tmpl2mp4(ts=ts,title=tmplname,debugTF=debugTF,_pn_=_pn_,tmplLst=tmplLst,lang=lang,ctrlSTRUCT=ctrlSTRUCT,outdir=outdir,dirname=dirname,**optx)
	return dpn

from _alan_optparse import parse_opt
def main_tst(description="Stock Update MP4 Creator",usage="Usage: %prog [option] daily_briefing|intraday_briefing [SYMBOL] [True|False]"):
	opts, args = parse_opt(sys.argv, description=description,usage=usage)
	ticker=getKeyVal(opts,'ticker','AAPL')
	txtOnly=getKeyVal(opts,'txtOnly',False)
	_pn_={}
	tmplname = 'daily_briefing' if len(args)<1 else args[0]
	ticker = ticker if len(args)<2 else args[1]
	txtOnly = True if len(args)>2 and args[2][:1].lower() in ['1','t'] else txtOnly
	sys.stderr.write("===RUN:{}:{}:{}\n".format(tmplname,ticker,txtOnly))
	dpn = tmpl_wrapper(tmplname,ticker=ticker,debugTF=True,txtOnly=txtOnly)
	return dpn


if __name__ == '__main__':
	main_tst()
