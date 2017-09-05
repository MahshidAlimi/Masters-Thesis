import os
import pandas as pd
import  numpy as np
import pickle

from utils import (get_market_ratings, get_financials_ticker, get_cyclical_adjustment,
                   transform_ratios, fill_out_weightings, calculate_score_rating,fill_missing_years,Yearly_Predictions,
                   Create_Yearly_Market_Data)

from years_ratio import calculate_ratios_yearly

_INPUT_DIR = os.getcwd() + '\\'

# prevent annoying pandas "A value is trying to be set on a copy of a slice" warnings
pd.options.mode.chained_assignment = None
hmm = True



# load in financials, market data, score lookup and cylical adjustments from pickle files
df_financials = pd.read_pickle('financials.pkl')
df_market_data = pd.read_pickle('market_data.pkl')
df_lookup = pd.read_pickle('ratings_lookup.pkl')
df_cyclical = pd.read_pickle('cyclical.pkl')

# only look at annual data for financials
df_financials = df_financials.query('ReportingPeriod == "YEARLY"')

_tickers = df_financials['tick_redux'].unique()
YEARLY_COL_NAMES = ['cash_conversion_three_year_mean',
                   'ebitda_interest_paid','cf_interest_paid', 'net_debt_pension_to_ebitda', 'net_debt_pension_to_ev',
                   'ltv', 'percentage_1_5','ret_on_assets','ret_on_equity','net_debt_eq','cash_ratio','debt_to_assets_ratio',
                   'debt_to_capital_ratio','debt_to_equity_ratio','tax_rate','ROC','interest_coverage_ratio','net_profit_margin',
                    'net_debt_debteq','total_debt_pension','net_debt_pension','pretax_income_margin','operation_margin',
                    'gross_margin','Ticker',"Country","Industry","Sub_Industry","Sales_Return"]

_financial_data = ['num_years_history','market_rating_average', 'market_rating_implied',
                   'num_average_rating_changes', 'num_implied_rating_changes', 'cyclical_adjustment',
                   'country', 'industry', 'industry_subgroup', 'revenue']


num_tickers = len(_tickers)
_counter = 0
test_df = pd.DataFrame(columns=YEARLY_COL_NAMES)
for _ticker in _tickers:     # _ticker = 'BN FP' # BN FP  SBRY LN
    # output progress
    print('Calculating values for ' + _ticker + ', {:.1%} complete'.format(_counter / num_tickers))
    # get financial data for company
    df_financials_sub, last_date = get_financials_ticker(df_financials, _ticker)
    # get cyclical adjustment
    cyclical_adjustment = get_cyclical_adjustment(df_financials_sub, df_cyclical)
    # is there a sufficient number of years history?
    num_years_history = len(df_financials_sub)
    if num_years_history > 1:
        # get ratio data
        yearly_ratio = calculate_ratios_yearly(df_financials_sub)
        yearly_df = fill_missing_years(yearly_ratio)
        company_yearly = [_ticker] * len(yearly_df)
        yearly_df["Ticker"] = company_yearly
        yearly_df["Country"] = [df_financials_sub['COUNTRY_FULL_NAME'].values[-1]] * len(yearly_df)
        yearly_df["Industry"] = [df_financials_sub['INDUSTRY_SECTOR'].values[-1]] * len(yearly_df)
        yearly_df["Sub_Industry"] = [df_financials_sub['INDUSTRY_SUBGROUP'].values[-1]] * len(yearly_df)
        yearly_df["Sales_Return"] = [df_financials_sub['SALES_REV_TURN'].values[-1]] * len(yearly_df)
        test_df = test_df.append(yearly_df)
    _counter += 1

test_df = test_df.reset_index()
test_df.rename(columns={'index':'date'},inplace = True)
if hmm == True:
    test_df,errors = Yearly_Predictions(test_df)
    test_df.to_pickle( 'HMM_Errors_mse.pkl')

else:
    market_yearly_df = Create_Yearly_Market_Data(test_df,df_market_data)
    market_yearly_df.to_pickle(_INPUT_DIR + 'market_yearly.pkl')


