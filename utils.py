from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from datetime import timedelta, date
import HMM

YEARLY_COL_NAMES = ['cash_conversion_three_year_mean',
                   'ebitda_interest_paid','cf_interest_paid', 'net_debt_pension_to_ebitda', 'net_debt_pension_to_ev', 'ltv',
                   'percentage_1_5','ret_on_assets','ret_on_equity','net_debt_eq','cash_ratio','debt_to_assets_ratio',
                   'debt_to_capital_ratio','debt_to_equity_ratio','tax_rate','ROC','interest_coverage_ratio','net_profit_margin',
                    'net_debt_debteq','total_debt_pension','net_debt_pension','pretax_income_margin','operation_margin',
                    'gross_margin']


RATIO_COL_NAMES = ['cagr', 'ebitda_margin', 'ebitda_margin_stability', 'cash_conversion_three_year_mean',
                   'ebitda_interest_paid','cf_interest_paid', 'net_debt_pension_to_ebitda', 'net_debt_pension_to_ev', 'ltv',
                   'percentage_1_5','ret_on_assets','ret_on_equity','net_debt_eq','cash_ratio','debt_to_assets_ratio',
                   'debt_to_capital_ratio','debt_to_equity_ratio','tax_rate','ROC','interest_coverage_ratio','net_profit_margin',
                   'net_debt_debteq','total_debt_pension','net_debt_pension','pretax_income_margin','operation_margin',
                    'gross_margin']
_RATIO_NAMES = ['10yr Revenue CAGR', 'EBITDA Margin', '10yr Margin Volatility (SD/AV)', 'Av. 3yr Cash Conversion',
                'EBITDA to Interest','Cash Flow to Interest', 'Net Debt & Pen : EBITDA', 'Net Debt & Pen / EV', 'LTV (GD/Pen to Tangibles)',
                '% Year 1-5 Debt Maturity','Return on Assets','Return on Equity','Net debt on Equity','Cash Ratio',
                'Debt to Assets Ratio','Debt to Capital Ratio','Debt to Equity Ratio','Rate on Tax','Return of Capital',
                'Interest Coverage %','Net profit %','Net debt to debt and Equity','Total debt from Pension',
                'Net Debt from Pension','Income Margin before Tax','Operation Income Margin','Gross Income Margin']
_WEIGHTINGS = [10, 10, 10, 10,10, 15, 15, 10, 10, 10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
_TYPES = ['P&L Metric', 'P&L Metric', 'P&L Metric', 'Cashflow Metric','Cashflow Metric', 'Cashflow Metric', 'Leverage Metric',
          'Leverage Metric', 'B/S Metric', 'B/S Metric','TBD Metric','TBD Metric','TBD Metric','TBD Metric',
          'TBD Metric','TBD Metric','TBD Metric','TBD Metric','TBD Metric','TBD Metric','TBD Metric','TBD Metric',
          'TBD Metric','TBD Metric','TBD Metric','TBD Metric','TBD Metric']

_SCORE_THRESHOLDS = [0, 10, 25, 40, 60, 80, 90]
RATING_BANDS = ['CCC', 'B', 'BB', 'BBB', 'A', 'AA', 'AAA']
_WEIGHTING_COLS = ['w_%s' % i for i in range(len(_WEIGHTINGS))]
RATIO_COLS = ['ratio_%s' % i for i in range(len(_RATIO_NAMES))]


def get_financials_ticker(_df_financials, _ticker):
    # get financial data for particular company spanning 10 years
    _df_sub = _df_financials.query('tick_redux == "%s"' % _ticker).sort_values('date')
    _df_sub.set_index('date', drop=True, inplace=True)
    # what is most recent date in financials data?
    _last_date = _df_sub.index.values[-1]
    return _df_sub, _last_date


def get_cyclical_adjustment(_df_financials_sub, _df_cyclical):
    # look up cyclical adjustment
    industry_subgroup = _df_financials_sub['INDUSTRY_SUBGROUP'].values[0]
    if industry_subgroup in _df_cyclical.index:
        return _df_cyclical.loc[industry_subgroup, 'SCORE ADJ. ']
    else:
        return 0.


def get_market_ratings(_df_market_data, _ticker, _last_date):
    # get market rating for first day after financial data came out
    _ten_years_ago = pd.to_datetime(_last_date.astype('M8[D]').astype('O') - relativedelta(years=10))
    df_market_ticker = _df_market_data.query('PRIMARY_EQUITY_TICKER == "%s"' % _ticker).query('AsAtDate > @_last_date')
    df_market_ticker_historical = _df_market_data.query('PRIMARY_EQUITY_TICKER == "%s"' % _ticker).query('AsAtDate <= @_last_date').query('AsAtDate >= @_ten_years_ago').dropna()
    num_average_rating_changes = (df_market_ticker_historical['AvRating'].shift(-1).fillna(method='ffill') != df_market_ticker_historical['AvRating']).sum()
    num_implied_rating_changes = (df_market_ticker_historical['ImpliedRating'].shift(-1).fillna(method='ffill') != df_market_ticker_historical['ImpliedRating']).sum()
    if not df_market_ticker.empty:
        df_market_sub = df_market_ticker.iloc[0, :]
        market_rating_average = df_market_sub.AvRating
        market_rating_implied = df_market_sub.ImpliedRating
        return market_rating_average, market_rating_implied, num_average_rating_changes, num_implied_rating_changes
    else:
        return np.nan, np.nan, np.nan, np.nan


def fill_out_weightings(_df_results, _w=_WEIGHTINGS):
    # fill out weightings when there are missing values
    # TODO this is terrible code and should be speeded up / improved
    for i in range(len(_WEIGHTINGS)):
        _df_results['w_%s' % i] = np.nan
    for _row_idx, _row in _df_results.iterrows():
        ratio_values = _row.loc[RATIO_COL_NAMES]
        if not ratio_values.isnull().any():
            _weightings = _w
        else:
            _data = ratio_values.values
            _df = pd.DataFrame({
                'value':_data,
                'weighting':_w,
                'type':_TYPES
            })
            _df.index = _RATIO_NAMES
            #_df = pd.DataFrame(np.array([_data, _w, _TYPES]), index=_RATIO_NAMES,
            #                   columns=['value', 'weighting', 'type'])
            _df.index.name = 'ratio'
            _df['value'] = _df['value'].astype(np.float)
            _df['weighting'] = _df['weighting'].astype(np.float)
            if not _df.groupby(['type']).sum()['value'].isnull().any():
                for _type in _df['type']:
                    _idx = np.where(_df['type'] == _type)
                    _values = _df['value'].iloc[_idx]
                    if _values.isnull().any():
                        _ideal_weight_sum = _df['weighting'].iloc[_idx].sum()
                        _weight_available = _df['weighting'].iloc[_idx] * (1. - _values.isnull())
                        _weight_available *= _ideal_weight_sum / _weight_available.sum()
                        _df['weighting'].iloc[_idx] = _weight_available
                _weightings = _df['weighting'].values
            else:
                _weightings = [np.nan] * len(_w)
        _df_results.loc[_row_idx, _WEIGHTING_COLS] = _weightings
    return _df_results


def calculate_score_rating(_df_results):
    # calculate model score and rating
    _ratio_values = _df_results.loc[:, RATIO_COLS].astype(float).fillna(0).values
    _weights = _df_results.loc[:, _WEIGHTING_COLS].astype(float).values
    _score = np.empty(len(_ratio_values)) * np.nan
    for i in range(len(_ratio_values)):
        _score[i] = np.sum(_ratio_values[i, :] * _weights[i, :]) / 5.
    _df_results['score'] = _score + _df_results['cyclical_adjustment']
    bad_idx = (_df_results[_df_results.loc[:, RATIO_COL_NAMES].isnull().all(axis=1)] == True).index
    _df_results['score_idx'] = np.digitize(_df_results['score'].astype(float).values, _SCORE_THRESHOLDS) - 1
    _df_results.loc[bad_idx, 'score'] = np.nan
    _df_results.loc[bad_idx, 'score_idx'] = np.nan
    _rating_dict = {0: 'CCC', 1: 'B', 2: 'BB', 3: 'BBB', 4: 'A', 5: 'AA', 6: 'AAA'}
    _df_results['model_rating'] = _df_results['score_idx'].map(_rating_dict)
    return _df_results


def transform_ratios(_df_results, _df_lookup):
    # transform ratio values to bucket
    df_ratios = _df_results.iloc[:, :len(RATIO_COL_NAMES)]
    _cols = df_ratios.columns
    for _idx in range(len(RATIO_COL_NAMES)):
        _values = df_ratios.loc[:, _cols[_idx]].astype(float)
        bad_idx = np.where(_values.isnull())[0]
        bins = _df_lookup.iloc[_idx, 1:].astype(float).values
        _ratings = np.digitize(_values.values, bins).astype(float)
        _ratings[bad_idx] = np.nan
        _df_results['ratio_%s' % _idx] = _ratings
    return _df_results


def rating_to_number(_df):
    # convert text ratings to their numerical equivalent
    for _col in ['model_rating', 'market_rating_average', 'market_rating_implied']:
        _rating_dict = {'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 6}
        _df[_col + '_number'] = _df[_col].map(_rating_dict)
    return _df


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days / 365)):
        yield start_date + timedelta(n) * 365

def fill_missing_years(ratios):
    #Fill missing years for yearly data, by finding the minimum and maximum dates of all the ratios.
    #And creating the data for those missing years.

    min_date = ratios[0].index[0]
    max_date = ratios[0].index[-1]
    for ratio in ratios:
        if ratio.index[0] < min_date:
            min_date = ratio.index[0]
        if ratio.index[1] > max_date:
            max_date = ratio.index[-1]

    difference_in_years = relativedelta(max_date, min_date).years
    cur_date = min_date
    date_range = []
    for i in range(difference_in_years):
        cur_date += timedelta(days=365)
        if cur_date.year % 4 == 0:
            cur_date += timedelta(days = 1)
        date_range.append(cur_date)


    yearly_df = pd.DataFrame(index = date_range)

    for i  in range(len(ratios)):
        for date in date_range:
            if date not in ratios[i].index:
                ratios[i][date] = ratios[i].mean()
        yearly_df[YEARLY_COL_NAMES[i]] = ratios[i]

    return  yearly_df



def Create_HMM_Predictions(yearly_df):
    #Create hidden markov model predictions. Calculates the transitions of the ratio, and predicts based on the next direction.

    predictions = []
    num_years = yearly_df.shape[1] - 1
    for i in range(len(yearly_df)):
        years = yearly_df.values[i]
        if np.nan in years:
            predictions.append(np.nan)
        else:
            try:
                direction_years = []
                for i in range(1, num_years):
                    if years[i] > years[i - 1]:
                        direction_years.append(1)
                    else:
                        direction_years.append(0)
                params = HMM.Prep_Forward(direction_years)
                probs = HMM.forward(params, np.array(direction_years))
                recent_year = probs[0][-1]
                direction = np.argmax(recent_year)
                diff = abs((years[-3] - years[-2]))
                if direction == 0:
                    prediction = years[-2] - diff
                else:
                    prediction = years[-2] + diff

                predictions.append(prediction)
            except:
                predictions.append(np.nan)

    real = yearly_df.values[:,-1]
    pred = np.array(predictions)
    error = ((real - pred) ** 2)/len(real)
    # error=mean_squared_error(real,pred)
    yearly_df["HMM_error"] = error

    return  yearly_df

def Yearly_Predictions(ratios_year_df):
    #Creates the HMM dataframe which replaces the value of the last year with the HMM prediction.

    no_index = ratios_year_df
    unique_tickers = no_index["Ticker"].unique()

    company_dfs = {}
    modified_df = None
    special_cols = ['Ticker','Country','Industry','Sub_Industry']
    errors = []
    for ticker in unique_tickers:
        print("Calculating HMM Predictions for " + ticker)
        ticker_df = no_index.loc[no_index.Ticker == ticker]
        ticker_years = ticker_df.date
        pivot = ticker_df.drop(["Ticker", "date","Country","Industry","Sub_Industry","Sales_Return"], axis=1).transpose()
        pivot.columns = [str(date.year) for date in ticker_years]
        pivot = Create_HMM_Predictions(pivot)
        pivot = pivot.transpose()
        pivot = pivot.reset_index()
        pivot.rename(columns = {'index':'date'},inplace = True)
        pivot['ticker'] = np.repeat(ticker,pivot.shape[0])
        if modified_df is None:
            modified_df = pivot
        else:
            modified_df = modified_df.append(pivot)

    return modified_df,errors


def nearest(dates, pivot):
    return abs(dates - pivot).idxmin()

def Create_Final_DF(df_matching_market,df_yearly):
    #Can this be improved into one loop?
    #Gets the market rating for each year by getting the rating nearest to the date of the ratio.
    df_final = None
    actual_dates = []
    for ticker in df_matching_market.PRIMARY_EQUITY_TICKER.unique():
        sub_df = df_matching_market.loc[df_matching_market.PRIMARY_EQUITY_TICKER == ticker]
        sub_yearly_df = df_yearly.loc[df_yearly.Ticker == ticker]
        closest_date = [sub_df.loc[nearest(sub_df.AsAtDate, date)].AsAtDate for date in sub_yearly_df.date]

        if df_final is None:
            df_final = sub_df.loc[sub_df.AsAtDate.isin(closest_date)]
        else:
            df_final = df_final.append(sub_df.loc[sub_df.AsAtDate.isin(closest_date)])

    for ticker in df_final.PRIMARY_EQUITY_TICKER.unique():
        sub_df = df_final.loc[df_final.PRIMARY_EQUITY_TICKER == ticker]
        sub_yearly_df = df_yearly.loc[df_yearly.Ticker == ticker]
        sub_dates = [sub_yearly_df.loc[nearest(sub_yearly_df.date, sub_date)].date for sub_date in sub_df.AsAtDate]
        actual_dates.extend(sub_dates)
    df_final['Actual'] = actual_dates
    return df_final


def Create_Rolling_Means(no_dupes):
    #Creates the rolling average for all ratios.
    rm_df = None
    for ticker in no_dupes.Ticker:
        sub_df = no_dupes.loc[no_dupes.Ticker == ticker].set_index('date')
        for column in sub_df.columns:
            sub_df[column + 'rm'] = sub_df[column].rolling(2).mean()

        if rm_df is None:
            rm_df = sub_df
        else:
            rm_df = rm_df.append(sub_df)

    rm_df.drop(['Tickerrm', 'Countryrm',
       'Industryrm', 'Sub_Industryrm', 'Sales_Returnrm', 'AvRatingrm',
       'ImpliedRatingrm'],axis =1,inplace = True)
    return rm_df

def Create_Yearly_Market_Data(df_yearly,df_market_data):
    #Combines the data by year with the yearly ratio data.
    yearly_tickers = df_yearly.Ticker.unique()
    market_tickers = df_market_data.PRIMARY_EQUITY_TICKER.unique()

    matching_tickers = [ticker for ticker in market_tickers if ticker in set(yearly_tickers)]
    df_matching_market = df_market_data.loc[df_market_data.PRIMARY_EQUITY_TICKER.isin(matching_tickers)]
    final_df = Create_Final_DF(df_matching_market,df_yearly)
    merged = df_yearly.merge(final_df, left_on=['date', 'Ticker'], right_on=['Actual', 'PRIMARY_EQUITY_TICKER'])
    merged.drop(['Actual', 'PRIMARY_EQUITY_TICKER', 'AsAtDate', 'Spread5y'], axis=1, inplace=True)
    no_dupes = merged.drop_duplicates()
    rm_df = Create_Rolling_Means(no_dupes)
    rm_final_df = rm_df[pd.notnull(rm_df['ROCrm'])]
    no_dupes_rm = rm_final_df.drop_duplicates()
    return  no_dupes_rm

