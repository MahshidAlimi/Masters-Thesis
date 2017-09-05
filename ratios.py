import pandas as pd
import numpy as np


def get_idx_years_back(_year, _num_years_back):
    # get indices for number years back
    _most_recent_year = _year.max()
    _years_back = _most_recent_year - _num_years_back
    if _year.min() > _years_back:
        return 0
    else:
        return np.max(np.where(_year <= _years_back)[0])


def get_most_recent(_values):
    # if most recent value is nan, use one before
    i = -1
    while _values[i] == np.nan:
        i-=1
    return _values[i]


def safe_divide(numerator, denominator):
    # slightly tortuous way to avoid dividing by nan or 0
    # Why not just add  a small denominator like 1e-8
    _bad_idx = np.isnan(denominator) | (denominator == 0.)
    denominator[_bad_idx] = 1.
    result = numerator / denominator
    result[_bad_idx] = np.nan
    return result

def calculate_CAGR(revenue,ten_years_back_idx):
    if revenue[ten_years_back_idx] == 0:
        return np.nan

    return  ((revenue[-1] / revenue[ten_years_back_idx]) ** 0.1) - 1.

def calculate_EBITDA(_df_financials_sub, nine_years_back_idx, revenue):
    ebitda_adjusted = _df_financials_sub['EBITDA_ADJUSTED']
    ebitda_adjusted = ebitda_adjusted.apply(lambda x: max(x, 0.001))
    ebitda_adjusted = ebitda_adjusted.fillna(_df_financials_sub['EBITDA']).values
    ebitda_margin_values = safe_divide(ebitda_adjusted[nine_years_back_idx:], revenue[nine_years_back_idx:])
    ebitda_margin = get_most_recent(ebitda_margin_values)

    # EBITDA Margin Stability
    if np.isnan(ebitda_margin_values).all():
        ebitda_margin_stability = np.nan
    else:
        ebitda_margin_stability = np.nanstd(ebitda_margin_values) / np.nanmean(ebitda_margin_values)

    return  ebitda_adjusted, ebitda_margin_values, ebitda_margin,ebitda_margin_stability


def calculate_CashConversion(cashflow_operating, assets_change, ebitda_adjusted,years_back_idx):
    cash_conversion = safe_divide(cashflow_operating + assets_change, ebitda_adjusted)
    cash_conversion_three_year_mean = np.max([cash_conversion[years_back_idx:].mean(), 0.])
    return  cash_conversion_three_year_mean

def calculate_INT_Expense(_df_financials_sub):
    interest_expense = _df_financials_sub['IS_INT_EXPENSE'].fillna(_df_financials_sub['IS_NET_INTEREST_EXPENSE'])
    interest_expense = interest_expense.fillna(_df_financials_sub['ARD_OTHER_FINANCIAL_LOSSES'])
    interest_expense = interest_expense.apply(lambda x: max(x, 0.1)).values
    return  interest_expense

def calculate_Interest(_df_financials_sub, value):
    interest_expense = calculate_INT_Expense(_df_financials_sub)
    interest_paid_values = value / interest_expense
    interest_paid = np.min([50., get_most_recent(interest_paid_values)])
    return  interest_paid,interest_paid_values



def calculate_NetPension(_df_financials_sub, ebitda_adjusted):
    total_debt = _df_financials_sub['SHORT_AND_LONG_TERM_DEBT']
    pension_liabilities = _df_financials_sub['PENSION_LIABILITIES'].fillna(_df_financials_sub['BS_PENSIONS_LT_LIABS'])
    cash = _df_financials_sub['CASH_AND_MARKETABLE_SECURITIES']
    subsidy_debt = _df_financials_sub['FINANCIAL_SUBSIDIARY_DEBT_&_ADJ']
    total_debt_pension = total_debt + pension_liabilities.fillna(0)
    net_debt_pension = total_debt_pension - cash - subsidy_debt.fillna(0)
    net_debt_pension_to_ebitda_values = net_debt_pension / ebitda_adjusted
    net_debt_pension_to_ebitda = get_most_recent(net_debt_pension_to_ebitda_values)

    if net_debt_pension_to_ebitda > 9.99:
        net_debt_pension_to_ebitda = np.nan
    elif net_debt_pension_to_ebitda < 0.:
        net_debt_pension_to_ebitda = 0.
    return  net_debt_pension_to_ebitda,net_debt_pension,total_debt_pension,net_debt_pension_to_ebitda_values

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
    net_debt_pension_to_ev = get_most_recent(net_debt_pension_to_ev_values)
    return  net_debt_pension_to_ev,net_debt_pension_to_ev_values


def calculate_LTV(_df_financials_sub,total_debt_pension):
    tangible_assets = _df_financials_sub['TANGIBLE_ASSETS']
    ltv_values = (total_debt_pension / tangible_assets)
    ltv = get_most_recent(ltv_values)
    return ltv

def calculate_1_5(_df_financials_sub, total_debt_pension):
    debt_2_5 = _df_financials_sub['BS_DEBT_SCHEDULE_YR_2_5'].values
    year_1_principal = _df_financials_sub['BS_YEAR_1_PRINCIPAL'].values
    debt_1_5 = debt_2_5 + year_1_principal
    percentage_1_5_values = debt_1_5 / total_debt_pension
    percentage_1_5 = get_most_recent(percentage_1_5_values)
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
    return  net_debt/(total_debt + _df_financials_sub["TOTAL_EQUITY"])


def get_rolling_mean(value):
    result = value.rolling(window=4,center=False).mean().dropna()
    if len(result) < 8:
        missing_values = [result.mean()] * (8 - len(result))
        result = missing_values + list(result)
    
    return result

def calculate_ratios(_df_financials_sub):
    # calculate all ratios from financials subset
    ratio_data = []

    # check that the data does actually span the last ten years
    year = pd.to_datetime(_df_financials_sub.index).year
    ten_years_back_idx = get_idx_years_back(year, 10)
    nine_years_back_idx = get_idx_years_back(year, 9)
    two_years_back_idx = get_idx_years_back(year, 2)
    # CAGR
    revenue = _df_financials_sub['SALES_REV_TURN'].values
    cagr = calculate_CAGR(revenue,ten_years_back_idx)
    ratio_data.append(cagr)

    # EBITDA
    ebitda_adjusted,ebitda_margin_values,\
    ebitda_margin, ebitda_margin_stability = calculate_EBITDA(_df_financials_sub,nine_years_back_idx,
                                                                          revenue)

    ratio_data.append(ebitda_margin)
    ratio_data.append(ebitda_margin_stability)
    # average 3 year Cash Conversion
    cash_conversion_three_year_mean = calculate_CashConversion(_df_financials_sub['CF_CASH_FROM_OPER'].values,
                                                               _df_financials_sub['CHG_IN_FXD_&_INTANG_AST_DETAILED'].values,
                                                               ebitda_adjusted,
                                                               two_years_back_idx)

    ratio_data.append(cash_conversion_three_year_mean)
    # EBITDA to Interest Paid
    ebitda_interest_paid,total_debt_pension = calculate_Interest(_df_financials_sub,ebitda_adjusted)

    ratio_data.append(ebitda_interest_paid)
    #Cash Flow to Interest Paid
    cf_interest_paid, total_debt_cf = calculate_Interest(_df_financials_sub, _df_financials_sub["CF_FREE_CASH_FLOW"])

    ratio_data.append(cf_interest_paid)
    # Net Debt and Pension to EBITDA
    net_debt_pension_to_ebitda,net_debt_pension,total_debt_pension,net_values = calculate_NetPension(_df_financials_sub,ebitda_adjusted)

    ratio_data.append(net_debt_pension_to_ebitda)
    ratio_data.extend(get_rolling_mean(net_values))

    # Net Debt & Pension to Enterprise Value
    net_debt_pension_to_ev,net_ev_values = calculate_NetEVPension(_df_financials_sub,net_debt_pension)

    ratio_data.append(net_debt_pension_to_ev)
    ratio_data.extend(get_rolling_mean(net_ev_values))
    # LTV
    ltv = calculate_LTV(_df_financials_sub,total_debt_pension)

    ratio_data.append(ltv)
    # 1 - 5 Year Percentage Debt Maturity
    percentage_1_5 = calculate_1_5(_df_financials_sub, total_debt_pension)

    ratio_data.append(percentage_1_5)
    #Return on Assets
    ret_on_assets = calculate_ret_on_assets(_df_financials_sub)

    ratio_data.append(get_most_recent(ret_on_assets))
    ratio_data.extend(get_rolling_mean(ret_on_assets))

    #Return on Equity
    ret_on_equity = _df_financials_sub["NET_INCOME"]/_df_financials_sub["TOTAL_EQUITY"]
    ratio_data.append(get_most_recent(ret_on_equity))
    ratio_data.extend(get_rolling_mean(ret_on_equity))

    #Net debt Equity
    net_debt_eq,total_liabilities = calculate_net_debt_equity(_df_financials_sub)
    ratio_data.append(get_most_recent(net_debt_eq))
    ratio_data.extend(get_rolling_mean(net_debt_eq))

    #Cash Ratio
    cash_ratio,total_debt = calculate_CashRatio(_df_financials_sub)
    ratio_data.append(get_most_recent(cash_ratio))
    ratio_data.extend(get_rolling_mean(cash_ratio))

    #Debt to assets ratio
    debt_to_assets_ratio = total_liabilities/_df_financials_sub["BS_TOT_ASSET"]
    ratio_data.append(get_most_recent(debt_to_assets_ratio))
    ratio_data.extend(get_rolling_mean(debt_to_assets_ratio))
    #Debt to Capital ratio
    debt_to_capital_ratio = total_debt / (total_debt  + _df_financials_sub["TOTAL_EQUITY"])
    ratio_data.append(get_most_recent(debt_to_capital_ratio))
    ratio_data.extend(get_rolling_mean(debt_to_capital_ratio))

    #Debt to Equity ratio
    debt_to_equity_ratio = total_debt / _df_financials_sub["TOTAL_EQUITY"]
    ratio_data.append(get_most_recent(debt_to_equity_ratio))
    ratio_data.extend(get_rolling_mean(debt_to_equity_ratio))

    #Tax rate
    tax_rate = _df_financials_sub['CF_CASH_PAID_FOR_TAX']/ _df_financials_sub['PRETAX_INC']
    ratio_data.append(get_most_recent(tax_rate))
    ratio_data.extend(get_rolling_mean(tax_rate))

    #Return on Capital
    ROC = (_df_financials_sub['EBITDA_ADJUSTED'] * (1 - tax_rate))/_df_financials_sub['TOTAL_INVESTED_CAPITAL']
    ratio_data.append(get_most_recent(ROC))
    ratio_data.extend(get_rolling_mean(ROC))

    #Interest Coverage ratio
    interest_expense = calculate_INT_Expense(_df_financials_sub)
    interest_coverage_ratio = _df_financials_sub["EBITDA_ADJUSTED"]  / interest_expense
    ratio_data.append(get_most_recent(interest_coverage_ratio))
    ratio_data.extend(get_rolling_mean(interest_coverage_ratio))


    #Net profit margin
    net_profit_margin = _df_financials_sub["NET_INCOME"] / revenue
    ratio_data.append(get_most_recent(net_profit_margin))
    ratio_data.extend(get_rolling_mean(net_profit_margin))


    #Net debt to debt and equity
    net_debt_debtequity = net_debt_to_debtandequity(_df_financials_sub)
    ratio_data.append(get_most_recent(net_debt_debtequity))
    ratio_data.extend(get_rolling_mean(net_debt_debtequity))
    #Total debt pension
    ratio_data.append(get_most_recent(total_debt_pension))
    ratio_data.extend(get_rolling_mean(total_debt_pension))
    #Net debt pension
    ratio_data.append(get_most_recent(net_debt_pension))
    ratio_data.extend(get_rolling_mean(net_debt_pension))

    #Pretax income margin ratio
    pretax_income_margin = _df_financials_sub['PRETAX_INC']/ revenue
    ratio_data.append(get_most_recent(pretax_income_margin))
    ratio_data.extend(get_rolling_mean(pretax_income_margin))

    #Operation Margin
    ratio_data.append(get_most_recent(_df_financials_sub["OPER_MARGIN"]))
    ratio_data.extend(get_rolling_mean(_df_financials_sub["OPER_MARGIN"]))
    #Gross Margin
    ratio_data.append(get_most_recent(_df_financials_sub["GROSS_MARGIN"]))
    ratio_data.extend(get_rolling_mean(_df_financials_sub["GROSS_MARGIN"]))
    # concatenate all ratios together
    #ratio_data = [cagr, ebitda_margin, ebitda_margin_stability, cash_conversion_three_year_mean, ebitda_interest_paid,
    #              net_debt_pension_to_ebitda, net_debt_pension_to_ev, ltv, percentage_1_5]

    print(len(ratio_data))
    return ratio_data
