import pandas as pd
import requests
import os
# Features to use: possession, timeouts_left_home, timeouts_left_away, score_difference, time_left_in_quarter, quarter, end.yardsToEndzone, end.down, end.distance, field_position_shift, type.id
def get_nfl_team_ids():
    """Fetch NFL team IDs and abbreviations from ESPN API"""
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    response = requests.get(url)
    data = response.json()
    
    team_dict = {}
    for team in data['sports'][0]['leagues'][0]['teams']:
        team_data = team['team']
        team_dict[team_data['id']] = {
            'abbreviation': team_data['abbreviation'],
            'name': team_data['displayName'],
            'short_name': team_data['shortDisplayName']
        }
    return team_dict


def add_timeouts(df, team_dict=None):
    """
    Adds timeout columns by properly identifying which team called each timeout.
    Uses ESPN team IDs if available, with fallback to possession data.
    Skips the first row (header row) when processing.
    """
    if team_dict is None:
        team_dict = get_nfl_team_ids()
    
    # Initialize
    home_team_id = int(df['home_team_id'].iloc[0])
    away_team_id = int(df['away_team_id'].iloc[0])
    home_timeouts = 3
    away_timeouts = 3
    
    home_timeouts_list = []
    away_timeouts_list = []
    
    # Add None for header row
    home_timeouts_list.append(None)
    away_timeouts_list.append(None)
    
    # Start processing from row 1 (skip header)
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        # Store current timeout counts
        home_timeouts_list.append(home_timeouts)
        away_timeouts_list.append(away_timeouts)
        
        if row['type.text'] == 'Timeout':
            timeout_text = str(row['text'])
            
            # Method 1: Check for explicit team abbreviation in timeout text
            if ' by ' in timeout_text:
                team_abbr = timeout_text.split(' by ')[1].split(' ')[0].upper()
                
                # Find which team this abbreviation belongs to
                abbr_found = False
                for team_id, abbreviations in team_dict.items():
                    can_break = False
                    # print(abbreviations, team_abbr)
                    for abbr in abbreviations:
                        if abbr == team_abbr:
                            if int(team_id) == int(home_team_id):
                                home_timeouts -= 1
                            else:
                                away_timeouts -= 1
                            abbr_found = True
                            can_break = True
                            break
                    if can_break:
                        break
                if not(abbr_found):
                    print(team_abbr)
            # Method 2: Fallback to possession team
            elif pd.notna(row.get('start.team.id')):
                if str(row['start.team.id']) == home_team_id:
                    home_timeouts -= 1  # offensive timeout
                else:
                    away_timeouts -= 1  # offensive timeout
        
        # Reset timeouts at half
        if row['period.number'] == 2 and row['type.text'] in ['End Period', 'End of Half']:
            home_timeouts = 3
            away_timeouts = 3
        home_timeouts = max(home_timeouts, 0)
        away_timeouts = max(away_timeouts, 0) 
    df['home_timeouts_left'] = home_timeouts_list
    df['away_timeouts_left'] = away_timeouts_list
    return df

def add_score_difference(df):
    '''
    Adds a column for score difference. A positive SD means home team is winning.
    Skips the first row (header row).
    '''
    score_diff = [None]  # None for header row
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        score_diff.append(row['homeScore'] - row['awayScore'])
    df['score_difference'] = score_diff
    return df

def add_possession_bool(df):
    '''
    Adds column to indicate whether home team has possession or not (using start.team.id)
    Skips the first row (header row).
    '''
    possession = [None]  # None for header row
    home_team_id = df.iloc[0]["home_team_id"]
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        possession.append(row['start.team.id'] == home_team_id)
    df['home_has_possession'] = possession
    return df

def add_time_left_in_seconds_for_period(df):
    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return None
        if ':' in time_str:
            minutes, seconds = map(int, time_str.split(':'))
            return (minutes * 60 + seconds) / 900
        return 0
    
    time_seconds = [None]  # None for header row
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        time_seconds.append(time_to_seconds(row['clock.displayValue']))
    df['time_left_in_period'] = time_seconds
    return df

def add_field_position_shift(df):
    '''
    How much the play was moved (field_position_shift = end.yardsToEndzone - start.yardsToEndzone)
    Skips the first row (header row).
    '''
    position_shift = [None]  # None for header row
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        position_shift.append(row['end.yardsToEndzone'] - row['start.yardsToEndzone'])
    df['field_position_shift'] = position_shift
    return df


def process_file(file_path, team_dict):
    '''
    Process a single CSV file in place
    '''
    df = pd.read_csv(file_path)
    
    # Apply all transformations
    df = add_timeouts(df, team_dict)
    df = add_score_difference(df)
    df = add_possession_bool(df)
    df = add_time_left_in_seconds_for_period(df)
    df = add_field_position_shift(df)
    df = add_final_score_difference(df)
    
    # Overwrite the original file
    df.to_csv(file_path, index=False)
    return True

def process_directory(input_dir, team_dict = None):
    """Process all CSV files in a directory in place"""
    if not(team_dict):
        team_dict = get_nfl_team_ids()
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            process_file(file_path, team_dict)
        print("Processed ", filename)


def extract_timeout_teams(df):
    """
    Extracts all unique team abbreviations used in timeout calls.
    Returns a set of team abbreviations found in timeout plays.
    """
    timeout_abbreviations = []
    
    for idx, row in df.iterrows():
        if row['type.text'] == 'Timeout':
            timeout_text = str(row['text'])
            if ' by ' in timeout_text:
                # Extract team abbreviation (e.g., "Timeout #1 by SEA at 12:00" → "SEA")
                team_abbr = timeout_text.split(' by ')[1].split(' ')[0].upper()
                timeout_abbreviations.append(team_abbr)
    
    return timeout_abbreviations


def has_overtime(file_path):
    """Check if a CSV file contains overtime plays"""
    try:
        df = pd.read_csv(file_path)
        return any(df['period.number'] > 4)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def delete_overtime_files(directory):
    """Delete all CSV files in directory that contain overtime plays"""
    deleted_files = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            if has_overtime(file_path):
                try:
                    os.remove(file_path)
                    deleted_files.append(filename)
                    print(f"Deleted: {filename}")
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")
    
    print(f"\nDeleted {len(deleted_files)} files containing overtime")
    if deleted_files:
        print("Deleted files:")
        for file in deleted_files:
            print(f"- {file}")

def ignore_overtime_periods(directory):
    """Delete overtime periods from all CSV files in a directory"""
    deleted_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            if has_overtime(file_path):
                df = pd.read_csv(file_path)
                df = df[(df["period.number"].isnull()) | (df["period.number"] == "") | (df["period.number"] <= 4)]
                df.to_csv(file_path, index=False)
                deleted_files.append(filename)
                print(f"Deleted overtime periods from {filename}")
    print(f"Deleted overtime periods from {len(deleted_files)} files")


def add_final_score_difference(df):
    '''
    Adds a column for final score difference. A positive SD means home team is winning.
    Skips the first row (header row).
    '''
    last_row = df.iloc[-1]
    final_score_diff = [None]  # None for header row
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        final_score_diff.append(last_row['homeScore'] - last_row['awayScore'])
    df['final_score_difference'] = final_score_diff
    return df


def add_relative_strength(df):
    '''
    Adds a column for relative strength.
    Skips the first row (header row).
    '''
    # relative strength is the home win probability of the home team at timestep 0 (first row)
    relative_strength = [df.iloc[1]['homeWinProbability']] * (len(df) - 1)
    relative_strength.insert(0, None)
    df['relative_strength'] = relative_strength
    return df


def has_overtime_improved(file_path):
    df = pd.read_csv(file_path)
    last_row = df.iloc[-1]
    if last_row['homeScore'] == last_row['awayScore']:
        return True
    return False

def get_overtime_files(directory):
    """Get all CSV files in directory that contain overtime plays"""
    deleted_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            if has_overtime(file_path) or has_overtime_improved(file_path): 
                print(f"Overtime: {filename}")

if __name__ == "__main__": 
    team_dict = {
    '22': ['ARI', 'ARZ'],
    '1': ['ATL'],
    '33': ['BAL', 'BLT'],
    '2': ['BUF'],
    '29': ['CAR'],
    '3': ['CHI'],
    '4': ['CIN', 'CIN.'],
    '5': ['CLE', 'CLV'],
    '6': ['DAL'],
    '7': ['DEN'],
    '8': ['DET'],
    '9': ['GB'],
    '34': ['HOU', 'HST'],
    '11': ['IND'],
    '30': ['JAX'],
    '12': ['KC'],
    '13': ['LV', 'OAK'],
    '24': ['LAC'],
    '14': ['LAR', 'LA'],
    '15': ['MIA'],
    '16': ['MIN'],
    '17': ['NE'],
    '18': ['NO'],
    '19': ['NYG'],
    '20': ['NYJ'],
    '21': ['PHI'],
    '23': ['PIT'],
    '25': ['SF'],
    '26': ['SEA'],
    '27': ['TB'],
    '10': ['TEN'],
    '28': ['WSH', 'WAS']
}
    # directories = ["dataset/2018", "dataset/2019", "dataset/2020", "dataset/2021", "dataset/2022", "dataset/2023", "dataset/2024"]
    # abbr = set()
    # for directory in directories:
    #     for filename in os.listdir(directory):
    #         if filename.endswith('.csv'):
    #             file_path = os.path.join(directory, filename)
    #             df = pd.read_csv(file_path)
    #             for team in extract_timeout_teams(df):
    #                 abbr.add(team)
    #     print(abbr)
    # print("FINAL:", abbr)
    directory = "dataset_interpolated_fixed/2024"
    # delete_overtime_files(directory)
    process_directory(directory, team_dict)
    get_overtime_files(directory)
    ignore_overtime_periods(directory)
    get_overtime_files(directory)