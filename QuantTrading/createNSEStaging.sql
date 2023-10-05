CREATE DATABASE nse;
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