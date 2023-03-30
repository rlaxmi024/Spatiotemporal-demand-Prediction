import utm
import pandas as pd


def to_utm(row):
    # Convert the start coordinates to utm
    try:
        x, y, _, _ = utm.from_latlon(row['start_lat'], row['start_lon'])
    except:
        x = y = float('nan')
    row['start_lat'], row['start_lon'] = x, y

    # Convert the end coordinates to utm
    try:
        x, y, _, _ = utm.from_latlon(row['end_lat'], row['end_lon'])
    except:
        x = y = float('nan')
    row['end_lat'], row['end_lon'] = x, y

    return row


# Define parameters
grid_size = 250
grid_start_lat = 39.92490
grid_start_long = -86.32736
grid_end_lat = 39.63890
grid_end_long = -85.95314

# Convert to UTM
grid_start_x, grid_start_y, _, _ = utm.from_latlon(grid_start_lat, grid_start_long)
grid_end_x, grid_end_y, _, _ = utm.from_latlon(grid_end_lat, grid_end_long)

# Get the maximum grid values
grid_max_x = (grid_end_x - grid_start_x) / grid_size
grid_max_y = (grid_start_y - grid_end_y) / grid_size

# Read the csv scooter dataset
df = pd.read_csv('data/purr_scooter_data.csv')
df.dropna(inplace=True)

# Drop the unneccessary columns
df.drop(columns=['trip_id', 'scooter_id', 'distance_miles'], inplace=True)

# Filter out the minutes and seconds since we need hour-level data
df['start_time_utc'] = df['start_time_utc'].str[: 13]
df['end_time_utc'] = df['end_time_utc'].str[: 13]

# Convert the lat long coordinates to utm
df = df.apply(to_utm, axis=1)
df.dropna(inplace=True)

# Convert the utm coordinates to grid indices
df['start_x'] = (df['start_lat'] - grid_start_x) / grid_size
df['start_y'] = (grid_start_y - df['start_lon']) / grid_size
df['end_x'] = (df['end_lat'] - grid_start_x) / grid_size
df['end_y'] = (grid_start_y - df['end_lon']) / grid_size

# Drop the unneccessary columns
df.drop(columns=['start_lat', 'start_lon', 'end_lat', 'end_lon'], inplace=True)

# Filter out trips outside Indianapolis
df = df.loc[
    (df['start_x'] >= 0) &
    (df['start_x'] <= grid_max_x) &
    (df['start_y'] >= 0) &
    (df['start_y'] <= grid_max_y) &
    (df['end_x'] >= 0) &
    (df['end_x'] <= grid_max_x) &
    (df['end_y'] >= 0) &
    (df['end_y'] <= grid_max_y)
]

# Convert to grid indices
df[['start_x', 'start_y', 'end_x', 'end_y']] = df[['start_x', 'start_y', 'end_x', 'end_y']].astype(int)

# Create dataframes for start and end times
start_df = df[['start_time_utc']]
end_df = df[['end_time_utc']]

# Convert the coordinates to a tuple format
start_df['xy'] = df[['start_x', 'start_y']]. apply(tuple, axis=1)
end_df['xy'] = df[['end_x', 'end_y']]. apply(tuple, axis=1)

# Add the count to get the demand
start_df['count'] = 1
end_df['count'] = 1

# Groupby the time and xy coordinates
start_df = start_df.groupby(['start_time_utc', 'xy']).count()
end_df = end_df.groupby(['end_time_utc', 'xy']).count()

# Loop through every unique hour timeslot and create the final preprocessed dataset
final_df = pd.DataFrame(columns=['time','grid_index','origin_count','destination_count'])
for hour in sorted(set(start_df.index.get_level_values(0)) | set(end_df.index.get_level_values(0))):
    dict = {}

    # If the hour has scooter trips that started in any grid index
    if hour in start_df.index:
        for row in start_df.loc[hour].itertuples():
            dict[row.Index] = {'origin_count': row.count, 'destination_count': 0}

    # If the hour has scooter trips that encded in any grid index
    if hour in end_df.index:
        for row in end_df.loc[hour].itertuples():
            if row.Index in dict:
                dict[row.Index]['destination_count'] = row.count
            else:
                dict[row.Index] = {'origin_count': 0, 'destination_count': row.count}

    # Store the demand matrix values for that hour
    for grid_index, count in dict.items():
        final_df.loc[len(final_df)] = [hour, grid_index, count['origin_count'], count['destination_count']]

# Store the final preprocessed dataset
final_df.to_csv('data/preprocessed_scooter_data.csv', index=False)