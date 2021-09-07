"""
TALAN
=====
A Technical Algorithmic Analytics Python Package
"""
__version__ = "0.1.0"
__author__ = 'Ted Hong'
__credits__ = 'Beyondbond Risk Lab'
import sys,os
extraDir=os.path.dirname(__file__)
sys.path.append(extraDir)
from _alan_date import *
from _alan_calc import *
from _alan_str import *
from _alan_date import *
from _alan_pppscf import batch_pppscf,vertex_locator,vertex_mnmx
from yh_chart import *
from yh_hist_batch import *
from yh_predefined import bb_predefined
from find_recent_eps import find_recent_eps
from headline_calc import find_hiloRecord,headline_calc
from headline_sts import headline_hist
from iex_peers import iex_peers,peers_performance,ticker2sectorRanking
from macro_event_yh import run_macro_event_yh
from plot_templates import plot_templates
from record_hilo import get_titlehead
from ticker2label import ticker2label
'''
import types
unLst=['x_','popen','Call','OLD','_DEPRECATED','Popen','__']
gDt = {k:v for (k,v) in globals().items() if callable(v) and not any(map(k.__contains__,unLst)) and isinstance(v,types.FunctionType)} 
fDt = {k:v for (k,v) in gDt.items() if globals()[k].__doc__ }
fLst=sorted(fDt.keys())
'''
fLst = ['batch_alan_calc', 'bb_predefined', 'calc_gkhv', 'calc_ma_diff', 'calc_macd', 'calc_rsi', 'check_latest_macro', 'chk_mnmx', 'combine_cmd2dict', 'conn2db', 'consecutive_sign', 'dLst2df', 'daily2month', 'delta2dates', 'df_output', 'df_tofile', 'dt2ymd', 'eten_minute', 'extra_opts', 'find_mdb', 'find_mnmx', 'find_numbers', 'fq2unit_str', 'freq_d2m', 'func2mdb', 'get2mndb', 'getCurrHist', 'getKeyVal', 'get_arg2func', 'get_csvdata', 'get_datax', 'get_globalmacrox', 'get_start_end', 'gsrg', 'headline_calc', 'hourly_eod', 'hw2ewma', 'import_module', 'insert_mdb', 'jobj2lsi', 'json_normalize', 'list2chunk', 'loc_aindex', 'loc_dindex', 'lst2dict', 'merge_t2l', 'merge_yqc', 'next_date', 'next_month_date', 'num2MP3', 'num_en2cn', 'opt_alan_calc', 'opt_yh_hist', 'pchg_calc', 'pdGroupMax', 'peers_performance', 'psd', 'pullStockHistory', 'qS_keyStatistics', 'qs_exec', 'qs_split', 'rdWord', 'remove_tags', 'renameDict', 'roundMP3', 'roundPct', 'roundSD', 'roundTBM', 'roundUSD', 'run_jj2', 'run_macd', 'run_macro_event_yh', 'run_ohlc', 'run_sma', 'run_tech', 's2dt', 'saferun', 'save2mndb', 'seOfDay', 'sqlQuery', 'step_index', 'stock_screener', 'str2float', 'str2value', 'str_contains', 'str_tofile', 'strc2float', 'subDF', 'subDict', 'subVDict', 'sub_special_char', 'syscall_eod', 'tg_latest2week', 'tg_next2week', 'ticker2sectorRanking', 'tmpl2lsi', 'trend_forecast', 'udfSentence', 'udfStr', 'udfWord', 'unique_set', 'upd2mdb', 'upd_temp2hist', 'upsert_mdb', 'useWeb', 'write2mdb', 'yh_batchRaw', 'yh_batchSpark', 'yh_financials', 'yh_financials_batch', 'yh_hist_query', 'yh_quote_comparison', 'yh_quote_curr', 'yh_rawdata', 'ymd2dt', 'ymd_delta', 'ymd_diff']
__all__ = fLst
