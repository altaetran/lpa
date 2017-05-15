import csv
import numpy as np
import pandas
import pandas_datareader as pdr
from pandas_datareader._utils import RemoteDataError
import datetime
import os

def get_dataset(filter_length=126):
    close_file = '../datasets/finance/finance_adj_close.csv'
    sector_file = '../datasets/finance/finance_sectors.csv'

    start_year = 2013
    end_year = 2017
    d1 = datetime.datetime(start_year, 1, 1)
    d2 = datetime.datetime(end_year, 1, 1)

    with open('../datasets/finance/companylist.csv') as f:
        reader = csv.reader(f, delimiter=',')
        counter = 0
        avail_stocks = {}
        stock_sectors = {}
        for line in reader:
            if counter >= 1:
                if line[5] != 'n/a' and int(line[5])>=start_year:
                    pass
                else:
                    symbol = line[0].strip()
                    avail_stocks[symbol] = line[1]
                    stock_sectors[symbol] = (line[6],line[7])
            counter += 1
            
        print('Number of companies: '+str(len(avail_stocks)))

    if not os.path.isfile(close_file):
        symbols, names = np.array(list(avail_stocks.items())).T

        quotes = []
        used_symbols = []
        counter = 0
        for symbol in symbols:
            try:
                quotes.append(pdr.get_data_yahoo(symbol,start=d1,end=d2))
                used_symbols.append(symbol)
            except RemoteDataError:
                print('>>Symbol: ' +symbol+' could not be read')
                pass
            counter += 1

        # Get correct number of time points
        n = int(np.median(zip(*map(np.shape, quotes))[0]))

        # Process the ones that don't have the right size
        valid_quotes = []
        valid_symbols = []
        valid_sectors = []
        for i in range(len(quotes)):
            if quotes[i].shape[0] == n:
                symbol = used_symbols[i]
                valid_quotes.append(quotes[i])
                valid_symbols.append(symbol)
                sect = stock_sectors[symbol]
                valid_sectors.append([symbol, sect[0], sect[1]])

        X = np.vstack([q['Adj Close'].as_matrix() for q in valid_quotes]).astype(np.float)

        with open(close_file, 'wb') as f:
            np.savetxt(f, X)

        with open(sector_file, 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            for row in valid_sectors:
                writer.writerow(row)    

    else:
        with open(close_file, 'r') as f:
            X = np.loadtxt(f)

        with open(sector_file, 'r') as f:
            valid_sectors = []
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                valid_sectors.append(row)

    X = np.log2(X)    
    ewma = np.vstack([pandas.stats.moments.ewma(X[i,:],halflife=filter_length) 
                      for i in range(X.shape[0])])
    
    print('Extracted '+str(X.shape[0])+' companies for analysis')
    
    X -= ewma
    m = X.shape[0]

    d = X[:,5:]-X[:,:-5]
    #rng = np.percentile(X,0.995,axis=1) - np.percentile(X,0.005,axis=1)
    
    keep_idx = np.max(np.abs(d),axis=1)<3

    valid_sectors = [valid_sectors[i] for i in range(m) if keep_idx[i]]
    X = X[keep_idx,:]

    print('Extracted '+str(X.shape[0])+' companies for further analysis')

    times = range(X.shape[1])

    n = X.shape[1]
    n_train = int(n*0.9) 
    n_val = int(n*0.05) 
    n_test = n-n_train-n_val

    X_train = X[:,:n_train]
    T_train = times[:n_train]
    X_val = X[:,n_train:n_train+n_val]
    T_val = times[n_train:n_train+n_val]
    X_test = X[:,n_train+n_val:]
    T_test = times[n_train+n_val:]

    return T_train,X_train,T_val,X_val,T_test,X_test,valid_sectors
        
