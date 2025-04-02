import ee
import geemap
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# Initialize the Earth Engine API
ee.Authenticate()
ee.Initialize(project='el-karter')

def create_roi(lon, lat, buffer_distance=200):
    point = ee.Geometry.Point([lon, lat])
    buffered = point.buffer(buffer_distance)
    return buffered.bounds()

def apply_scale_and_offset_l4_5_7(image):
    st_band = image.select('ST_B6').multiply(0.00341802).add(149.0).rename('surface_temp')
    return image.addBands(st_band)

def apply_scale_and_offset_l8_9(image):
    st_band = image.select('ST_B10').multiply(0.00341802).add(149.0).rename('surface_temp')
    return image.addBands(st_band)

def create_cloud_mask_l4_5_7(image):
    qa = image.select('QA_PIXEL')
    fill_mask = qa.bitwiseAnd(1).eq(0)                  # 0 = not fill, 1 = fill
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)            # 0 = not cloud, 1 = cloud
    cloud_dilated_mask = qa.bitwiseAnd(1 << 1).eq(0)    # 0 = not dilated cloud, 1 = dilated cloud
    cloud_shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)     # 0 = not cloud shadow, 1 = cloud shadow
    snow_mask = qa.bitwiseAnd(1 << 5).eq(0)             # 0 = not snow, 1 = snow
    mask = fill_mask.And(cloud_mask).And(cloud_dilated_mask).And(cloud_shadow_mask).And(snow_mask)
    return mask

def create_cloud_mask_l8_9(image):
    qa = image.select('QA_PIXEL')
    fill_mask = qa.bitwiseAnd(1).eq(0)                  # 0 = not fill, 1 = fill
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)            # 0 = not cloud, 1 = cloud
    cloud_dilated_mask = qa.bitwiseAnd(1 << 1).eq(0)    # 0 = not dilated cloud, 1 = dilated cloud
    cirrus_mask = qa.bitwiseAnd(1 << 2).eq(0)           # 0 = not cirrus, 1 = cirrus
    cloud_shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)     # 0 = not cloud shadow, 1 = cloud shadow
    snow_mask = qa.bitwiseAnd(1 << 5).eq(0)             # 0 = not snow, 1 = snow
    mask = fill_mask.And(cloud_mask).And(cloud_dilated_mask).And(cirrus_mask).And(cloud_shadow_mask).And(snow_mask)
    return mask

def process_landsat(collection, roi, satellite_name, apply_scale_func, create_mask_func):
    filtered = collection.filterBounds(roi)
    count = filtered.size().getInfo()
    print(f"Number of {satellite_name} scenes: {count}")
    results = []
    image_list = filtered.toList(filtered.size())

    for i in range(count):
        try:
            image = ee.Image(image_list.get(i))                                             # Get the image
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()   # retrieve date
            scaled_image = apply_scale_func(image)                                          # Apply scale and offset
            mask = create_mask_func(image)                                                  # Apply cloud mask
            masked_image = scaled_image.updateMask(mask)
            stats = masked_image.select('surface_temp').reduceRegion(
                reducer=ee.Reducer.count(), geometry=roi, scale=30).getInfo()
            
            # If there are valid pixels, compute the average temperature
            if stats['surface_temp'] > 0:
                mean_temp = masked_image.select('surface_temp').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=roi, scale=30).getInfo()['surface_temp']
                
                results.append({'satellite': satellite_name, 'date': date, 'temperature': mean_temp})
                print(f"{satellite_name} - {date}: {mean_temp:.3f} K")

        except Exception as e:
            print(f"Error processing {satellite_name} image {i}: {e}")
    
    return results

def LST_analysis(lon, lat, output_csv='lake_temperature.csv'):
    roi = create_roi(lon, lat)

    collections = [
        ('LANDSAT/LT04/C02/T1_L2', 'Landsat 4', apply_scale_and_offset_l4_5_7, create_cloud_mask_l4_5_7),
        ('LANDSAT/LT05/C02/T1_L2', 'Landsat 5', apply_scale_and_offset_l4_5_7, create_cloud_mask_l4_5_7),
        # ('LANDSAT/LE07/C02/T1_L2', 'Landsat 7', apply_scale_and_offset_l4_5_7, create_cloud_mask_l4_5_7),
        # ('LANDSAT/LC08/C02/T1_L2', 'Landsat 8', apply_scale_and_offset_l8_9, create_cloud_mask_l8_9),
        # ('LANDSAT/LC09/C02/T1_L2', 'Landsat 9', apply_scale_and_offset_l8_9, create_cloud_mask_l8_9)
        ]

    results = []
    for collection, name, scale_func, cloud_mask_func in collections:
        img_col = ee.ImageCollection(collection)
        results = process_landsat(img_col, roi, name, scale_func, cloud_mask_func)
        results.extend(results)

    data_df = pd.DataFrame(results)
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df.sort_values('date', inplace=True)
    data_df.to_csv(output_csv, index=False)

    return data_df

def LST_trends_analysis(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    annual_avg = df.groupby('year')['temperature'].mean().reset_index()
    monthly_avg = df.groupby('month')['temperature'].mean().reset_index()

    seasonal_pattern = {
        'monthly_avg': monthly_avg,
        'max_month': monthly_avg.loc[monthly_avg['temperature'].idxmax(), 'month'],
        'min_month': monthly_avg.loc[monthly_avg['temperature'].idxmin(), 'month'],
        'change_rate': monthly_avg['temperature'].max() - monthly_avg['temperature'].min()}

    monthly_avg_yearly = df.groupby(['year', 'month'])['temperature'].mean().reset_index()
    annual_extremes = df.groupby('year')['temperature'].agg(annual_max='max', annual_min='min').reset_index()

    results_dict = {
        'annual_avg': annual_avg,
        'seasonal_pattern': seasonal_pattern,
        'monthly_avg_yearly': monthly_avg_yearly,
        'annual_extremes': annual_extremes}
    
    return results_dict

def trends_report(df):
    # Annual statistics
    annual = df.groupby(df['date'].dt.year).agg(
        Mean_Temp=('temperature', 'mean'),
        Min_Temp=('temperature', 'min'),
        Max_Temp=('temperature', 'max'),
        Observations=('temperature', 'count')).reset_index()

    # Trend analysis
    years = annual['date'].values
    temps = annual['Mean_Temp'].values
    slope, _, r_value, p_value, _ = stats.linregress(years, temps)
    
    baseline = annual.loc[annual['date'].between(1984, 2000), 'Mean_Temp'].mean()
    recent = annual.loc[annual['date'] >= 2001, 'Mean_Temp'].mean()

    threshold_high = df['temperature'].quantile(0.9)
    threshold_low = df['temperature'].quantile(0.1)
    
    report_dict = {
        'period': f"{years.min()}-{years.max()}",
        'trend_K_decade': slope * 10,
        'p_value': p_value,
        'r_squared': r_value**2,
        'total_warming_K': temps[-1] - temps[0],
        'baseline_mean_K': baseline,
        'recent_mean_K': recent,
        'anomaly_K': recent - baseline,
        'extreme_heat_threshold_K': threshold_high,
        'extreme_cold_threshold_K': threshold_low}
    
    return report_dict, annual

def create_interactive_map(lon, lat, buffer_distance=200):
    center_point = ee.Geometry.Point([lon, lat])
    circular_buffer = center_point.buffer(buffer_distance)
    bounds = circular_buffer.bounds()

    m = geemap.Map()
    m.centerObject(center_point, 16)
    m.addLayer(circular_buffer, {'color': 'red'}, 'Circular Buffer')
    m.addLayer(bounds, {'color': 'blue'}, 'Buffer Bounds')
    m.addLayer(center_point, {'color': 'yellow'}, 'Center Point')
    return m

def plot_time_series(df, figsize=(14, 7), title='Lake Surface Temperature Time Series', window=20):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    satellites = df['satellite'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(satellites)))

    for i, sat in enumerate(satellites):
        sat_data = df[df['satellite'] == sat]
        ax.scatter(sat_data['date'], sat_data['temperature'], label=f'{sat} (n={len(sat_data)})', 
                   color=colors[i], edgecolor='w', linewidth=0.5)

    df_sorted = df.sort_values('date')
    df_sorted['moving_avg'] = df_sorted['temperature'].rolling(window=window, min_periods=1, center=True).mean()
    ax.plot(df_sorted['date'], df_sorted['moving_avg'], color='k', label=f'Moving Avg trend over {window} days', zorder=10)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, which='both', linestyle=':', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.02, 1),  loc='upper left', borderaxespad=0, frameon=True, facecolor='white')
    
    total_obs = len(df)
    ax.annotate(f'Total observations: {total_obs}', xy=(0.02, 0.95), xycoords='axes fraction',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    return fig

def plot_annual_avg(df):
    df['year'] = df['date'].dt.year
    annual_avg = df.groupby('year')['temperature'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    plt.plot(annual_avg['year'], annual_avg['temperature'],
             marker='o', linestyle='-', color='green')
    plt.xlabel('Year')
    plt.ylabel('Temperature (K)')
    plt.title('Annual Average Temperature')
    
    start_year = int(annual_avg['year'].min())
    end_year = int(annual_avg['year'].max())

    plt.xticks(range(start_year, end_year + 1, 5))
    plt.minorticks_on()
    plt.grid(True, which='major', linewidth=1.0)
    plt.grid(True, which='minor', linewidth=0.5, linestyle=':')
    plt.tight_layout()
    plt.show()

def plot_monthly_avg(df):
    df['month'] = df['date'].dt.month
    monthly_avg = df.groupby('month')['temperature'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_avg['month'], monthly_avg['temperature'], marker='s', linestyle='-', color='purple')
    plt.xlabel('Month')
    plt.ylabel('Temperature (K)')
    plt.title('Overall Monthly Average Temperature')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_histogram_density(df):
    plt.figure(figsize=(10, 5))
    plt.hist(df['temperature'], bins=30, density=True, alpha=0.6, color='orange', edgecolor='black')
    df['temperature'].plot(kind='density', color='darkblue', linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Density')
    plt.title('Histogram and Density Plot of Temperature')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():

    lon, lat = 33.1684, -0.9147
    df = LST_analysis(lon, lat)
    trends = LST_trends_analysis(df)

    m = create_interactive_map(lon, lat)
    m.to_html('lake_map.html')
    
    plot_time_series(df)
    plot_annual_avg(df)
    plot_monthly_avg(df)
    plot_histogram_density(df)

    # Access annual averages
    print("Annual Averages:")
    print(trends['annual_avg'])

    # Access seasonal pattern
    print("\nSeasonal Pattern:")
    print(trends['seasonal_pattern']['monthly_avg'])  
    print("\nSeasonal statistics:")
    print("Warmest month:", trends['seasonal_pattern']['max_month'])
    print("Coldest month:", trends['seasonal_pattern']['min_month'])
    print("Seasonal change:", trends['seasonal_pattern']['change_rate'])

    # Access extreme events
    print("\nExtreme Events:")
    print("Yearly extremes:\n", trends['annual_extremes'])

    print(f"\nAnalysis Period: {df['date'].dt.year.min()} to {df['date'].dt.year.max()}")
    print(f"Total Observations: {len(df)}")
    print(f"Satellites Included: {df['satellite'].unique().tolist()}")

    report, annual = trends_report(df)
    print(report)
   

if __name__ == "__main__":
    main()
