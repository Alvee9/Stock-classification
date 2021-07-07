from datapackage import Package

package = Package('https://datahub.io/core/s-and-p-500-companies-financials/datapackage.json')

def fetch_company_names():
    companies = []
    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            r = resource.read()
            cnt = 0
            for i in r:
                cnt += 1
                companies.append(i[0])
    return list(dict.fromkeys(companies))

companies = fetch_company_names()

print(len(companies))



import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials

def fetch_time_series(company):
    yahoo_financials = YahooFinancials(company)

    data = yahoo_financials.get_historical_price_data(start_date='2017-01-01', 
                                                    end_date='2019-12-31', 
                                                    time_interval='daily')
    comp_df = pd.DataFrame(data[company]['prices'])
    comp_df = comp_df.drop('date', axis=1).set_index('formatted_date')
    # print(comp_df)
    comp_df.to_csv("Comp_time_series/{}.csv".format(company))

def fetch_time_series_yfinance(company):
    comp_df = yf.download(company, 
    start='2017-01-01', 
    end='2019-12-31', 
    progress=False,
    interval='1d')
    if comp_df.empty:
        return
    comp_df.to_csv("Comp_time_series/{}.csv".format(company))

for company in companies:
    print(company)
    fetch_time_series_yfinance(company)




