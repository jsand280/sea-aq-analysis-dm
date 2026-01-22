import requests
import pandas as pd
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Get a new free API key at: https://explore.openaq.org/register
API_KEY = os.environ.get("OPENAQ_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://api.openaq.org/v3"

HEADERS = {
    "X-API-Key": API_KEY,
    "Accept": "application/json"
}

CITIES = {
    'Bangkok': {'lat': 13.7563, 'lon': 100.5018},
    'Ho Chi Minh City': {'lat': 10.8231, 'lon': 106.6297},
    'Kuala Lumpur': {'lat': 3.1390, 'lon': 101.6869},
    'Singapore': {'lat': 1.3521, 'lon': 103.8198}
}

PM25_PARAMETER_ID = 2

def get_locations_near_city(city_name, lat, lon, radius=25000):
    url = f"{BASE_URL}/locations"
    params = {
        'coordinates': f"{lat},{lon}",
        'radius': radius,
        'limit': 100
    }

    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    locations = []
    for loc in data.get('results', []):
        pm25_sensors = []
        for sensor in loc.get('sensors', []):
            param = sensor.get('parameter', {})
            if param.get('id') == PM25_PARAMETER_ID or param.get('name') == 'pm25':
                pm25_sensors.append({
                    'sensor_id': sensor['id'],
                    'sensor_name': sensor.get('name', 'PM2.5')
                })

        if pm25_sensors:
            locations.append({
                'location_id': loc['id'],
                'location_name': loc.get('name', 'Unknown'),
                'latitude': loc['coordinates']['latitude'],
                'longitude': loc['coordinates']['longitude'],
                'is_monitor': loc.get('isMonitor', False),
                'is_mobile': loc.get('isMobile', False),
                'sensors': pm25_sensors
            })

    print(f"  Found {len(locations)} locations with PM2.5 sensors near {city_name}")
    return locations

def get_sensor_hourly_data(sensor_id, date_from, date_to):
    url = f"{BASE_URL}/sensors/{sensor_id}/hours"
    all_data = []
    page = 1

    while True:
        params = {
            'datetime_from': date_from,
            'datetime_to': date_to,
            'limit': 1000,
            'page': page
        }

        response = requests.get(url, headers=HEADERS, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        results = data.get('results', [])
        if not results:
            break

        for r in results:
            period = r.get('period', {})
            datetime_from = period.get('datetimeFrom', {}).get('utc')

            all_data.append({
                'datetime': datetime_from,
                'value': r.get('value'),
                'sensor_id': sensor_id
            })

        meta = data.get('meta', {})
        found_str = str(meta.get('found', '0'))
        # API returns '>1000' when count exceeds 1000, so handle that case
        if found_str.startswith('>'):
            found = int(found_str[1:]) + 1  # Ensure we continue paginating
        else:
            found = int(found_str) if found_str.isdigit() else 0

        if page * 1000 >= found or len(results) < 1000:
            break

        page += 1
        time.sleep(0.2)

    return all_data

def download_city_data(city_name, city_info, year=2023):
    print(f"\nDownloading data for {city_name}...")

    locations = get_locations_near_city(city_name, city_info['lat'], city_info['lon'])

    if not locations:
        raise ValueError(f"No PM2.5 monitoring locations found for {city_name}")

    stationary_monitors = [loc for loc in locations if loc['is_monitor'] and not loc['is_mobile']]
    locations_to_use = stationary_monitors[:5] if stationary_monitors else locations[:5]
    print(f"  Using {len(locations_to_use)} locations")

    date_from = f"{year}-01-01T00:00:00Z"
    date_to = f"{year}-12-31T23:59:59Z"

    all_measurements = []

    for loc in locations_to_use:
        print(f"  Fetching from: {loc['location_name']} (ID: {loc['location_id']})")

        for sensor in loc['sensors']:
            sensor_id = sensor['sensor_id']
            print(f"    Sensor {sensor_id}...", end=" ")

            measurements = get_sensor_hourly_data(sensor_id, date_from, date_to)

            for m in measurements:
                m['city'] = city_name
                m['location_name'] = loc['location_name']
                m['location_id'] = loc['location_id']

            all_measurements.extend(measurements)
            print(f"{len(measurements)} records")
            time.sleep(0.3)

    if not all_measurements:
        raise ValueError(f"No measurements retrieved for {city_name}")

    df = pd.DataFrame(all_measurements)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.rename(columns={'value': 'pm25'})
    df = df.dropna(subset=['pm25'])
    df = df[df['pm25'] >= 0]

    df_hourly = df.groupby([pd.Grouper(key='datetime', freq='h'), 'city']).agg({
        'pm25': 'mean'
    }).reset_index()

    print(f"  Total: {len(df_hourly)} hourly records for {city_name}")
    return df_hourly

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("="*60)
    print("OpenAQ v3 API Data Download")
    print("="*60)
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-8:] if len(API_KEY) > 16 else '***'}")
    print(f"Target year: 2023")
    print(f"Cities: {', '.join(CITIES.keys())}")

    for city_name, city_info in CITIES.items():
        df = download_city_data(city_name, city_info, year=2023)

        output_file = os.path.join(RAW_DIR, f"{city_name.lower().replace(' ', '_')}_pm25.csv")
        df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")

        time.sleep(1)

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
