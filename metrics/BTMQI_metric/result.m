
clear;
clc;

load result
[abs(corr(TMID_btmqi,TMID_mos,'type','spearman')); ...
 abs(corr(TMID2015_btmqi,TMID2015_mos,'type','spearman'))]
