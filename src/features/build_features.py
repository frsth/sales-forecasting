import pandas as pd
import numpy as np
import datetime
import holidays

def build_features(df):
    daily_df = (df.groupby('datetime').agg(total_sales=('Sales', 'sum'), avg_discount=('Discount', 'mean')).reset_index())

    # Fill NaN avg discounts
    daily_df['avg_discount'] = daily_df['avg_discount'].fillna(0)

    # Fill all days
    all_days_range = pd.date_range(start=daily_df['datetime'].min(), end=daily_df['datetime'].max()) 
    all_days = pd.DataFrame({'datetime': all_days_range})
    # Merge with your daily data
    daily_df = all_days.merge(daily_df, on='datetime', how='left')

    # Replace NaN (days with no sales) by 0
    daily_df['total_sales'] = daily_df['total_sales'].fillna(0)

    # year
    daily_df['year'] = daily_df['datetime'].dt.year

    # month
    daily_df['month'] = daily_df['datetime'].dt.month
    daily_df['month_sin'] = np.sin(2 * np.pi * daily_df['month'] / 12)
    daily_df['month_cos'] = np.cos(2 * np.pi * daily_df['month'] / 12)
    daily_df.drop('month', axis = 1, inplace = True)

    # week
    daily_df['week_of_month'] = daily_df['datetime'].apply(lambda d: (d.day - 1) // 7 + 1)
    daily_df['week_of_month_sin'] = np.sin(2 * np.pi * daily_df['week_of_month'] / 12)
    daily_df['week_of_month_cos'] = np.cos(2 * np.pi * daily_df['week_of_month'] / 12)
    daily_df.drop('week_of_month', axis = 1, inplace = True)

    # day of week
    daily_df['day_of_week'] = daily_df['datetime'].dt.dayofweek 
    daily_df['day_of_week_sin'] = np.sin(2 * np.pi * daily_df['day_of_week'] / 12)
    daily_df['day_of_week_cos'] = np.cos(2 * np.pi * daily_df['day_of_week'] / 12)
    daily_df.drop('day_of_week', axis = 1, inplace = True)

    # is discount
    daily_df['is_discount'] = daily_df['avg_discount'] > 0


    # holiday
    us_holidays  = holidays.CountryHoliday('US', years=range(2013, 2020))
    canada_holidays = holidays.CountryHoliday('CA', years=range(2013, 2020))
    holiday_dates =list(canada_holidays.keys())
    holiday_dates.append(list(us_holidays.keys()))
    daily_df['is_holiday'] = daily_df['datetime'].dt.date.isin(holiday_dates)

    # lag1
    daily_df['lag1'] = daily_df['total_sales'].shift(1)

    # lag7
    daily_df['lag7'] = daily_df['total_sales'].shift(7)

    
    return daily_df


def build_features_test_data(df):
    daily_df = (df.groupby('datetime').agg(total_sales=('Sales', 'sum'), avg_discount=('Discount', 'mean')).reset_index())

    # Fill all days
    all_days_range = pd.date_range(start=daily_df['datetime'].min(), end=daily_df['datetime'].max()) 
    all_days = pd.DataFrame({'datetime': all_days_range})
    # Merge with your daily data
    daily_df = all_days.merge(daily_df, on='datetime', how='left')

    # Replace NaN (days with no sales) by 0
    daily_df['total_sales'] = daily_df['total_sales'].fillna(0)

    # year
    daily_df['year'] = daily_df['datetime'].dt.year

    # month
    daily_df['month'] = daily_df['datetime'].dt.month
    daily_df['month_sin'] = np.sin(2 * np.pi * daily_df['month'] / 12)
    daily_df['month_cos'] = np.cos(2 * np.pi * daily_df['month'] / 12)
    daily_df.drop('month', axis = 1, inplace = True)

    # week
    daily_df['week_of_month'] = daily_df['datetime'].apply(lambda d: (d.day - 1) // 7 + 1)
    daily_df['week_of_month_sin'] = np.sin(2 * np.pi * daily_df['week_of_month'] / 12)
    daily_df['week_of_month_cos'] = np.cos(2 * np.pi * daily_df['week_of_month'] / 12)
    daily_df.drop('week_of_month', axis = 1, inplace = True)

    # day of week
    daily_df['day_of_week'] = daily_df['datetime'].dt.dayofweek 
    daily_df['day_of_week_sin'] = np.sin(2 * np.pi * daily_df['day_of_week'] / 12)
    daily_df['day_of_week_cos'] = np.cos(2 * np.pi * daily_df['day_of_week'] / 12)
    daily_df.drop('day_of_week', axis = 1, inplace = True)

    # is discount
    daily_df['is_discount'] = daily_df['avg_discount'] > 0


    # holiday
    us_holidays  = holidays.CountryHoliday('US', years=range(2013, 2020))
    canada_holidays = holidays.CountryHoliday('CA', years=range(2013, 2020))
    holiday_dates =list(canada_holidays.keys())
    holiday_dates.append(list(us_holidays.keys()))
    daily_df['is_holiday'] = daily_df['datetime'].dt.date.isin(holiday_dates)
    
    return daily_df
    