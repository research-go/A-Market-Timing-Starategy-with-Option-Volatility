import pandas as pd
import openpyxl
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

n_30 = 30*24*60*60
n_365 = 365*24*60*60
error_margin = 1e-4
data_clean_error_margin_ratio = 1
data_clean_error_margin = 0.20
# Data From Excels


def excel_adj(data):
    data.drop([0, 1, 2], inplace=True)
    data = data.set_index('Date')
    return data


data_put = pd.read_csv('put.csv')
data_put['date'] = [dt.datetime.strptime(str(x), "%Y%m%d") for x in data_put['date']]
data_put = data_put.set_index('date')
data_call = pd.read_csv('call.csv')
data_call['date'] = [dt.datetime.strptime(str(x), "%Y%m%d") for x in data_call['date']]
data_call = data_call.set_index('date')
data_cd91 = pd.read_excel('CD91.xlsx')
data_cd91.drop([0, 1, 2], inplace=True)
data_cd91 = data_cd91.iloc[:, 0:2]
data_cd91.columns = ['Date', 'Last Price']
data_cd91 = data_cd91.set_index('Date')


def strike_adj(s):
    if s % 5 == 0:
        return s
    else:
        return s + 0.5


def month_adj(m):
    if m == 10:
        return 'A'
    if m == 11:
        return 'B'
    if m == 12:
        return 'C'
    return m


class Option:
    def __init__(self, equity, strike, risk_free, maturity):
        self.eqt = equity
        self.stk = strike
        self.rf = risk_free
        self.mat = maturity

    def pricing(self, volatility, is_call=True):
        def formula(d1, d2):
            return self.eqt*norm.cdf(d1) - self.stk*np.exp(-self.rf*self.mat)*norm.cdf(d2)
        tmp1 = (np.log(self.eqt/self.stk) + (self.rf + (volatility**2)/2)*self.mat)/(volatility*(self.mat**(1/2)))
        tmp2 = (np.log(self.eqt/self.stk) + (self.rf - (volatility**2)/2)*self.mat)/(volatility*(self.mat**(1/2)))
        factor = int(is_call)*2-1  # call or put option
        price = factor*formula(factor*tmp1, factor*tmp2)
        return price

    def implied_volatility(self, market_price, iv_initial=10, error_limit=1E-6):  # Newton-Raphson
        def slope(v1):
            return self.eqt*((self.mat/(2*np.pi))**(1/2))*np.exp(-(((np.log(self.eqt/self.stk)+(self.rf+(v1**2)/2)*self.mat)/(v1*(self.mat**(1/2))))**2)/2)
        iv = iv_initial
        error = self.pricing(iv) - market_price
        while abs(error) > error_limit:
            iv = iv - error/slope(iv)
            error = self.pricing(iv) - market_price
        return float(iv)


class Vkospi:
    def __init__(self, date: dt.datetime):
        self.date = date
        self.option_due = None
        self.n_t = None

    def nth_weekday(self, nth_week=2, week_day=3):
        if self.date.year == 2024 and self.date.month == 8:
            return dt.datetime(2024, 8, 8, 0, 0, 0)
        if self.date.year == 2024 and self.date.month == 9:
            return dt.datetime(2024, 9, 12, 0, 0, 0) # base case
        temp = self.date.replace(day=1)
        adj = (week_day - temp.weekday()) % 7
        temp += dt.timedelta(days=adj)
        temp += dt.timedelta(weeks=nth_week-1)
        return max(filter(lambda x: x <= temp, data_put.index))  # trading day adjusted:

    def get_recent_due(self):
        # get 2nd thursday of the same month
        this_month_due_date = self.nth_weekday()
        # in case today already passed the due date (10/15) -> get next month due date
        if self.date < this_month_due_date:
            return this_month_due_date
        elif self.date >= this_month_due_date:
            if self.date.month == 12:
                next_month = dt.datetime(self.date.year+1, 1, 1, 0, 0, 0)
            else:
                next_month = dt.datetime(self.date.year, self.date.month+1, 1, 0, 0, 0)
            return Vkospi(next_month).nth_weekday()

    def get_next_due(self):
        tmp = self.get_recent_due()
        return Vkospi(tmp).get_recent_due()

    def risk_free(self):
        return data_cd91.loc[max(filter(lambda x: x < self.date, data_cd91.index)), 'Last Price']

    def volatility(self, position='both', maturity='this'):
        if maturity == 'this':
            self.option_due = self.get_recent_due()
        elif maturity == 'next':
            self.option_due = self.get_next_due()
        self.n_t = int((self.option_due - self.date).total_seconds())
        t = self.n_t/n_365
        diff_min = 10000
        iteration = ((self.option_due.year-2019)*12 + self.option_due.month - 1)*117 + 1
        # 풋옵션과 콜옵션 가격 차이가 최소인 행사가격을 찾는다.
        for i in range(iteration, iteration+117):
            col_call = data_call.columns[i]
            col_put = data_put.columns[i]
            diff = abs(round(data_call.loc[self.date, col_call] - data_put.loc[self.date, col_put], 2))
            if np.isnan(diff):
                continue
            diff_min = min(diff, diff_min)
        diff_min_list = []
        for i in range(iteration, iteration+117):
            col_call = data_call.columns[i]
            col_put = data_put.columns[i]
            diff = abs(round(data_call.loc[self.date, col_call] - data_put.loc[self.date, col_put], 2))
            if abs(diff - diff_min) <= error_margin:
                diff_min_list.append((col_call, col_put))
        option_price_diff = 0
        # 그런 행사가격이 복수일 경우
        if len(diff_min_list) > 1:
            diff_min = 10000
            # KOSPI200과의 차이가 최소인 행사가격을 찾는다.
            for i in diff_min_list:
                diff_min = min(abs(data_put.loc[self.date, 'kospi200'] - strike_adj(int(i[0][-3:]))), diff_min)
            strike_max = 0
            # 그런 행사가격 중 최대값을 찾는다.
            for i in diff_min_list:
                if abs(abs(data_put.loc[self.date, 'kospi200'] - strike_adj(int(i[0][-3:]))) - diff_min) <= error_margin:
                    strike_max = max(strike_adj(int(i[0][-3:])), strike_max)
            f = 0
            for i in diff_min_list:
                if strike_adj(int(i[0][-3:])) == strike_max:
                    option_price_diff = data_call.loc[self.date, i[0]] - data_put.loc[self.date, i[1]]
                    f = strike_max + np.exp(t*self.risk_free()/100)*option_price_diff
                    break
        elif len(diff_min_list) == 0:
            print(self.date)
            return np.nan
        else:
            option_price_diff = data_call.loc[self.date, diff_min_list[0][0]] - data_put.loc[self.date, diff_min_list[0][1]]
            f = strike_adj(int(diff_min_list[0][0][-3:])) + np.exp(t*self.risk_free()/100)*option_price_diff
        if option_price_diff >= 0:
            k0 = max(filter(lambda x: x <= f, [2.5*i for i in range(200)]))
        else:
            k0 = min(filter(lambda x: x >= f, [2.5*i for i in range(200)]))
        sum_strike_call = 0
        sum_strike_put = 0
        cnt_call = 0
        cnt_put = 0
        min_price_call = 0
        min_price_put = 0
        print(k0)
        for i in range(iteration, iteration+117):
            col = data_call.columns[i]
            if strike_adj(int(col[-3:])) > k0:
                if np.isnan(data_call.loc[self.date, col]):
                    break
                if cnt_call == 3:
                    min_price_call = 1
                # Data Clean Check
                before = data_call.loc[self.date, data_call.columns[max(i-1, iteration)]]
                if not np.isnan(before) and min_price_call == 0:
                    if before*(1+data_clean_error_margin_ratio) < data_call.loc[self.date, col] and\
                            before+data_clean_error_margin < data_call.loc[self.date, col]:
                        print(col)
                        return np.nan
                if data_call.loc[self.date, col] == 0.01:
                    cnt_call += 1
                else:
                    cnt_call = 0
                q = data_call.loc[self.date, col]
                if min_price_call == 1:
                    q = 0.01
                sum_strike_call += 2.5/((strike_adj(int(col[-3:])))**2)*q

        for i in range(1, 118):
            col = data_put.columns[iteration+117 - i]
            if strike_adj(int(col[-3:])) < k0:
                if np.isnan(data_put.loc[self.date, col]):
                    break
                if cnt_put == 3:
                    min_price_put = 1
                # Data Clean Check
                before = data_put.loc[self.date, data_put.columns[min(iteration+117 - i + 1, iteration+117-1)]]
                if not np.isnan(before) and min_price_put == 0:
                    if before*(1+data_clean_error_margin_ratio) < data_put.loc[self.date, col] and\
                            before + data_clean_error_margin < data_put.loc[self.date, col]:
                        print(col)
                        return np.nan

                if data_put.loc[self.date, col] == 0.01:
                    cnt_put += 1
                else:
                    cnt_put = 0
                q = data_put.loc[self.date, col]
                if min_price_put == 1:
                    q = 0.01
                sum_strike_put += 2.5/((strike_adj(int(col[-3:])))**2)*q
        sum_adjustment = 0
        if position == 'call':
            sum_adjustment = sum_strike_call
        elif position == 'put':
            sum_adjustment = sum_strike_put
        elif position == 'both':
            sum_adjustment = sum_strike_call + sum_strike_put
        vol = 2/t*np.exp(self.risk_free()*t)*sum_adjustment - 1/t*((f/k0 - 1)**2)
        return vol

    def index(self, position='both'):
        tmp = Vkospi(self.date)
        vol1 = tmp.volatility(maturity='this', position=position)
        n_t1 = tmp.n_t
        t1 = n_t1/n_365
        if n_t1 > 2_592_000:
            return 100*(vol1**(1/2))
        else:
            vol2 = tmp.volatility(maturity='next', position=position)
            n_t2 = tmp.n_t
            t2 = n_t2/n_365
            return round(100*(((t1*vol1*(n_t2 - n_30)/(n_t2 - n_t1) + t2*vol2*(n_30 - n_t1)/(n_t2 - n_t1))*n_365/n_30)**(1/2)), 2)


if __name__ == "__main__":
    # test1 = Vkospi(dt.datetime.strptime("2021-10-19", "%Y-%m-%d"))
    # test2 = Vkospi(dt.datetime.strptime("2021-10-20", "%Y-%m-%d"))
    test3 = Vkospi(dt.datetime.strptime("2024-07-30", "%Y-%m-%d"))
    print(test3.volatility())


