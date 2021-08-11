import pandas as pd
import numpy as np

class Dataset:
    def __init__(self) -> None:
        self.companies = []
    
    def read_company_names(self):
        with open('../Dataset/s&p500_symbols.txt') as f:
            lines = f.read().split()
            for l in lines:
                self.companies.append(l)
        return self.companies

    def get_closing_prices(self, companies, number_of_companies):
        data = []
        for idx, company in enumerate(companies):
            # df = pd.read_csv('Dataset/Comp_time_series/{}.csv'.format(company))
            try:
                df = pd.read_csv('../Dataset/Comp_time_series/{}.csv'.format(company))
                if len(df['Close']) != 753:
                    continue
                data.append([idx] + list(df['Close']))
            except FileNotFoundError:
                # print(company, 'Not found')
                pass
            if len(data) == number_of_companies:
                break
        return data

    def calc_price_changes(self, companies, number_of_companies):
        data = []
        for idx, company in enumerate(companies):
            try:
                df = pd.read_csv('../Dataset/Comp_time_series/{}.csv'.format(company))
                if len(df['Close']) != 753:
                    continue
            except FileNotFoundError:
                # print(company, 'Not found')
                continue
            price_changes = [idx] + [0] * len(df['Close'])
            for i in range(1, len(df['Close'])):
                price_changes[i] = (df['Close'][i] - df['Close'][i - 1]) / df['Close'][i - 1] * 100
            data.append(price_changes)

            if len(data) == number_of_companies:
                break
        return data


    def average_ndays(self, data, n):
        new_data = []
        for trendline in data:
            sum = 0.0
            new_trendline = [trendline[0]]
            for i in range(1, len(trendline)):
                sum += trendline[i]
                divisor = i
                if i > n:
                    divisor = n
                    sum -= trendline[i - n]
                new_trendline.append(sum / divisor)
            new_data.append(new_trendline)
        return new_data


    def normalize(self, data):
        new_data = []
        for trendline in data:
            p0 = trendline[1]
            new_trendline = [trendline[0]]
            for i in range(1, len(trendline)):
                pt = trendline[i]
                px = (pt - p0) / p0 * 100 + 100
                new_trendline.append(px)
            new_data.append(new_trendline)
        return new_data
