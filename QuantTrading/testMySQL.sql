USE nse;

create TABLE cmStaging(
    symbol VARCHAR(256),
    series VARCHAR(256),
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    last FLOAT,
    prevclose FLOAT,
    tottrdqty FLOAT,
    tottrdval FLOAT,
    timestamp date,
    totaltrades FLOAT,
    isin VARCHAR(256)
)


Load data infile 'E:\Eskills-Academy-projects\\tutorials\\Masters-DataSciences\\QuantTrading\\data\\cm01JAN2018bhav.csv' into table cmStaging fields terminated by ',' ignore 1 lines
(symbol, series, open, high, low, close, last, prevclose, tottrdqty,tottrdval, @timestamp, totaltrades, isin)
SET timestamp = STR_TO_DATE(@timestamp, '%d-%b-%Y');


DROP TABLE cmstaging;