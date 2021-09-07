#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from optparse import OptionParser

def subDict(myDict,kyLst,reverseTF=False):
	"""
	return a sub-dict of myDict based on the matching [kyLst] keys
	OR
	sub-dict based on the non-matching [kyLst] keys
	"""
	if reverseTF is True: # invert-match, select non-matching [kyLst] keys
		return { ky:myDict[ky] for ky in myDict.keys() if ky not in kyLst }
	else:
		return { ky:myDict[ky] for ky in myDict.keys() if ky in kyLst }

def qs_split(xstr,d1='&',d2='='):
	""" split query string into a dict object """
	d = {k:v for k,v in [lx.split(d2) for lx in xstr.split(d1)]}
	return d

def qs_exec(xstr):
	""" convert commend string into a dict object """
	d = {}
	exec(xstr,globals(),d)
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

def parse_opt(argv=[],retParser=False,**kwargs):
	""" command-line options initial setup
	    Arguments:
		argv:	list arguments, usually passed from sys.argv
		retParser:	OptionParser class return flag, default to False
	    Return: (options, args) tuple if retParser is False else OptionParser class 
	"""
	optx = dict(usage="usage: %prog [option] SYMBOL1 ...", version="%prog 0.7",
		description="ALAN utilities")
	optx.update({ky:kwargs[ky] for ky in kwargs.keys() if ky in optx.keys()})
	parser = OptionParser(**optx)
	parser.add_option("-f","--file",action="store",dest="filename",
		help="input FILENAME")
	parser.add_option("-s","--sep",action="store",dest="sep",default="|",
		help="field separator (default: |)")
	parser.add_option("-d","--database",action="store",dest="dbname",default="ara",
		help="database DBNAME (default: ara)")
	parser.add_option("","--host",action="store",dest="hostname",default="localhost",
		help="database HOST (default: localhost)")
	parser.add_option("-t","--table",action="store",dest="tablename",
		help="database TABLENAME")
	parser.add_option("","--no_database_save",action="store_false",dest="saveDB",default=True,
		help="no save to database (default: save)")
	parser.add_option("","--start",action="store",dest="start",
		help="START date in YYYYMMDD")
	parser.add_option("","--end",action="store",dest="end",
		help="END date in YYYYMMDD")
	days = int(kwargs['days']) if 'days' in kwargs else 730
	parser.add_option("","--days",action="store",dest="days",default=days,type=int,
		help="number of days from END date (default: {})".format(days))
	parser.add_option("-o","--output",action="store",dest="output",
		help="output type [json|csv|html] (default: None)")
	parser.add_option("-l","--lang",action="store",dest="lang",default="cn",
		help="language (default: cn)")
	parser.add_option("","--extra_js",action="store",dest="extraJS",
		help="extra JSON in DICT format.")
	parser.add_option("","--extra_qs",action="store",dest="extraQS",
		help="extra GET string format like k1=v1&k2=v2; ")
	parser.add_option("","--extra_xs",action="store",dest="extraXS",
		help="extra excutable string like k1=v1;k2=v2; ")
	parser.add_option("","--debug",action="store_true",dest="debugTF",default=False,
		help="debugging (default: False)")
	(options, args) = parser.parse_args(argv[1:])
	try:
		opts = vars(options)
		extra_opts(opts,xkey='extraJS',method='JS',updTF=True)
		extra_opts(opts,xkey='extraQS',method='QS',updTF=True)
		extra_opts(opts,xkey='extraXS',method='XS',updTF=True)
		opts.update(args=args,narg=len(args))
	except Exception as e:
		sys.stderr.write(str(e)+"\n")
	if retParser is True:
		return (opts, parser)
	return (opts, args)

if __name__ == '__main__':
	(opts, args)=parse_opt(sys.argv)
	if (opts['debugTF']==True):
		sys.stderr.write("{} {}\n".format(opts, args))
