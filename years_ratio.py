import pandas as pd
import numpy as np

#This is almos the same asa the ratios file, but it gets the data for all years, instead of the most recent one.


def safe_divide(numerator, denominator):
    # slightly tortuous way to avoid dividing by nan or 0
    _bad_idx = np.isnan(denominator) | (denominator == 0.)
    denominator[_bad_idx] = 1.
    result = numerator / denominator
    result[_bad_idx] = np.nan
    return result

def calculate_CAGR(revenue,ten_years_back_idx):
    if revenue[ten_years_back_idx] == 0:
        return np.nan

    return  ((revenue[-1] / revenue[ten_years_back_idx]) ** 0.1) - 1.

def calculate_EBITDA(_df_financials_sub, revenue):
    ebitda_adjusted = _df_financials_sub['EBITDA_ADJUSTED']
    ebitda_adjusted = ebitda_adjusted.apply(lambda x: max(x, 0.001))
    ebitda_adjusted = ebitda_adjusted.fillna(_df_financials_sub['EBITDA'])
    ebitda_margin_values = safe_divide(ebitda_adjusted, revenue)
    ebitda_margin = ebitda_margin_values

    # EBITDA Margin Stability

    ebitda_margin_stability = ebitda_margin_values.std() / ebitda_margin_values.mean()


    return  ebitda_adjusted, ebitda_margin_values, ebitda_margin,ebitda_margin_stability


def calculate_CashConversion(cashflow_operating, assets_change, ebitda_adjusted):
    cash_conversion = safe_divide(cashflow_operating + assets_change, ebitda_adjusted)
    #cash_conversion_three_year_mean = np.max([cash_conversion.mean(), 0.])
    return  cash_conversion

def calculate_INT_Expense(_df_financials_sub):
    interest_expense = _df_financials_sub['IS_INT_EXPENSE'].fillna(_df_financials_sub['IS_NET_INTEREST_EXPENSE'])
    interest_expense = interest_expense.fillna(_df_financials_sub['ARD_OTHER_FINANCIAL_LOSSES'])
    interest_expense = interest_expense.apply(lambda x: max(x, 0.1)).values
    return  interest_expense

def calculate_Interest(_df_financials_sub, value):
    interest_expense = calculate_INT_Expense(_df_financials_sub)
    interest_paid_values = value / interest_expense
    temp_df = pd.DataFrame(index=interest_paid_values.index)
    temp_df["interest"] = [min(50.0,interest) for interest in interest_paid_values]
    return  temp_df.interest,interest_paid_values



def calculate_NetPension(_df_financials_sub, ebitda_adjusted):
    total_debt = _df_financials_sub['SHORT_AND_LONG_TERM_DEBT']
    pension_liabilities = _df_financials_sub['PENSION_LIABILITIES'].fillna(_df_financials_sub['BS_PENSIONS_LT_LIABS'])
    cash = _df_financials_sub['CASH_AND_MARKETABLE_SECURITIES']
    subsidy_debt = _df_financials_sub['FINANCIAL_SUBSIDIARY_DEBT_&_ADJ']
    total_debt_pension = total_debt + pension_liabilities.fillna(0)
    net_debt_pension = total_debt_pension - cash - subsidy_debt.fillna(0)
    net_debt_pension_to_ebitda_values = net_debt_pension / ebitda_adjusted
    net_debt_pension_to_ebitda = net_debt_pension_to_ebitda_values

    temp_df = pd.DataFrame(index=net_debt_pension_to_ebitda.index)
    temp_df["net_pension"]= [np.nan if pension > 9.99 else 0. if pension < 0. else pension for pension
                                  in net_debt_pension_to_ebitda]


    return  temp_df["net_pension"],net_debt_pension,total_debt_pension

def calculate_CashRatio(_df_financials_sub):
    total_debt = _df_financials_sub['SHORT_AND_LONG_TERM_DEBT']
    pension_liabilities = _df_financials_sub['PENSION_LIABILITIES'].fillna(_df_financials_sub['BS_PENSIONS_LT_LIABS'])
    cash = _df_financials_sub['CASH_AND_MARKETABLE_SECURITIES']
    subsidy_debt = _df_financials_sub['FINANCIAL_SUBSIDIARY_DEBT_&_ADJ'].fillna(0)
    total_debt_pension = total_debt + pension_liabilities.fillna(0)
    cashratio = cash/total_debt_pension
    return  cashratio,total_debt

def calculate_NetEVPension(_df_financials_sub,net_debt_pension):
    market_cap = _df_financials_sub['HISTORICAL_MARKET_CAP'].values
    enterprise_value = market_cap + net_debt_pension
    net_debt_pension_to_ev_values = net_debt_pension / enterprise_value
    net_debt_pension_to_ev = net_debt_pension_to_ev_values
    return  net_debt_pension_to_ev


def calculate_LTV(_df_financials_sub,total_debt_pension):
    tangible_assets = _df_financials_sub['TANGIBLE_ASSETS']
    ltv_values = (total_debt_pension / tangible_assets)
    ltv = ltv_values
    return ltv

def calculate_1_5(_df_financials_sub, total_debt_pension):
    debt_2_5 = _df_financials_sub['BS_DEBT_SCHEDULE_YR_2_5'].values
    year_1_principal = _df_financials_sub['BS_YEAR_1_PRINCIPAL'].values
    debt_1_5 = debt_2_5 + year_1_principal
    percentage_1_5_values = debt_1_5 / total_debt_pension
    percentage_1_5 = percentage_1_5_values
    return  percentage_1_5

def calculate_ret_on_assets(_df_financials_sub):
    return _df_financials_sub["NET_INCOME"]/_df_financials_sub["BS_TOT_ASSET"]

def calculate_net_debt_equity(_df_financials_sub):
    total_liabilities = _df_financials_sub["BS_TOT_ASSET"] - _df_financials_sub["TOTAL_EQUITY"]
    net_debt_eq = total_liabilities / _df_financials_sub["TOTAL_EQUITY"]
    return net_debt_eq,total_liabilities

def net_debt_to_debtandequity(_df_financials_sub):
    net_debt = _df_financials_sub["NET_DEBT"].values
    total_debt = _df_financials_sub["SHORT_AND_LONG_TERM_DEBT"].values
    return  (net_debt/(total_debt + _df_financials_sub["TOTAL_EQUITY"]))


def calculate_ratios_yearly(_df_financials_sub):
    # calculate all ratios from financials subset
    ratio_data = []

    # check that the data does actually span the last ten years
    year = pd.to_datetime(_df_financials_sub.index).year
    #ten_years_back_idx = get_idx_years_back(year, 10)
    #nine_years_back_idx = get_idx_years_back(year, 9)
    #two_years_back_idx = get_idx_years_back(year, 2)

    # CAGR
    revenue = _df_financials_sub['SALES_REV_TURN'].values
    #cagr = calculate_CAGR(revenue)
   # ratio_data.append(cagr)

    # EBITDA
    ebitda_adjusted,ebitda_margin_values,\
    ebitda_margin, ebitda_margin_stability = calculate_EBITDA(_df_financials_sub,
                                                                          revenue)

    #ratio_data.append(ebitda_margin_stability)
    # average 3 year Cash Conversion
    cash_conversion= calculate_CashConversion(_df_financials_sub['CF_CASH_FROM_OPER'].values,
                                                               _df_financials_sub['CHG_IN_FXD_&_INTANG_AST_DETAILED'].values,
                                                               ebitda_adjusted)

    ratio_data.append(cash_conversion)
    # EBITDA to Interest Paid
    ebitda_interest_paid,total_debt_pension = calculate_Interest(_df_financials_sub,ebitda_adjusted)

    ratio_data.append(ebitda_interest_paid)
    #Cash Flow to Interest Paid
    cf_interest_paid, total_debt_cf = calculate_Interest(_df_financials_sub, _df_financials_sub["CF_FREE_CASH_FLOW"])

    ratio_data.append(cf_interest_paid)
    # Net Debt and Pension to EBITDA
    net_debt_pension_to_ebitda,net_debt_pension,total_debt_pension = calculate_NetPension(_df_financials_sub,ebitda_adjusted)

    ratio_data.append(net_debt_pension_to_ebitda)
    # Net Debt & Pension to Enterprise Value
    net_debt_pension_to_ev = calculate_NetEVPension(_df_financials_sub,net_debt_pension)

    ratio_data.append(net_debt_pension_to_ev)
    # LTV
    ltv = calculate_LTV(_df_financials_sub,total_debt_pension)

    ratio_data.append(ltv)
    # 1 - 5 Year Percentage Debt Maturity
    percentage_1_5 = calculate_1_5(_df_financials_sub, total_debt_pension)

    ratio_data.append(percentage_1_5)
    ret_on_assets = calculate_ret_on_assets(_df_financials_sub)
    ratio_data.append(ret_on_assets.fillna(np.nanmean(ret_on_assets)))


    ret_on_equity = _df_financials_sub["NET_INCOME"]/_df_financials_sub["TOTAL_EQUITY"]
    ratio_data.append(ret_on_equity)


    net_debt_eq,total_liabilities = calculate_net_debt_equity(_df_financials_sub)
    ratio_data.append(net_debt_eq.fillna(np.nanmean(net_debt_eq)))

    cash_ratio,total_debt = calculate_CashRatio(_df_financials_sub)
    ratio_data.append(cash_ratio.fillna(np.nanmean(cash_ratio)))

    debt_to_assets_ratio = total_liabilities/_df_financials_sub["BS_TOT_ASSET"]
    ratio_data.append(debt_to_assets_ratio.fillna(np.nanmean(debt_to_assets_ratio)))

    debt_to_capital_ratio = total_debt / (total_debt  + _df_financials_sub["TOTAL_EQUITY"])
    ratio_data.append(debt_to_capital_ratio.fillna(np.nanmean(debt_to_capital_ratio)))

    debt_to_equity_ratio = total_debt / _df_financials_sub["TOTAL_EQUITY"]
    ratio_data.append(debt_to_equity_ratio.fillna(np.nanmean(debt_to_equity_ratio)))

    tax_rate = _df_financials_sub['CF_CASH_PAID_FOR_TAX']/ _df_financials_sub['PRETAX_INC']
    ratio_data.append(tax_rate.fillna(np.nanmean(net_debt_eq)))

    ROC = (_df_financials_sub['EBITDA_ADJUSTED'] * (1 - tax_rate))/_df_financials_sub['TOTAL_INVESTED_CAPITAL']
    ratio_data.append(ROC.fillna(np.nanmean(ROC)))


    interest_expense = calculate_INT_Expense(_df_financials_sub)
    interest_coverage_ratio = _df_financials_sub["EBITDA_ADJUSTED"]  / interest_expense
    ratio_data.append(interest_coverage_ratio.fillna(np.nanmean(interest_coverage_ratio)))

    net_profit_margin = _df_financials_sub["NET_INCOME"] / revenue
    ratio_data.append(net_profit_margin.fillna(np.nanmean(net_profit_margin)))

    net_debt_debtequity = net_debt_to_debtandequity(_df_financials_sub)
    ratio_data.append(net_debt_debtequity)

    ratio_data.append((total_debt_pension))
    ratio_data.append((net_debt_pension))

    pretax_income_margin = _df_financials_sub['PRETAX_INC'] / revenue
    ratio_data.append((pretax_income_margin))

    ratio_data.append((_df_financials_sub["OPER_MARGIN"]))
    ratio_data.append((_df_financials_sub["GROSS_MARGIN"]))


    # concatenate all ratios together
    #ratio_data = [cagr, ebitda_margin, ebitda_margin_stability, cash_conversion_three_year_mean, ebitda_interest_paid,
    #              net_debt_pension_to_ebitda, net_debt_pension_to_ev, ltv, percentage_1_5]
    return ratio_data
