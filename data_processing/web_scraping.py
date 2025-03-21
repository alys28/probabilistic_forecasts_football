import requests
from typing import List, Tuple, Set, Dict, Any
import pandas as pd
import os

def getIDs(year: int) -> List[Tuple[str, int, str, str]]:
    """
    Fetches unique match data for NFL teams for the given years, including match ID, home win status, and team IDs.

    Args:
        year (int): year for which to fetch the match data.

    Returns:
        List[Tuple[str, int, str, str]]: A list of unique tuples, where each tuple contains:
                                         - Match ID (str)
                                         - Home win status (1 if home team won, 0 otherwise)
                                         - Home team ID (str)
                                         - Away team ID (str)
    """
    base_team_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    team_schedule_url_template = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{TEAM_ID}/schedule?season={YEAR}"

    # Fetch all team IDs
    response = requests.get(base_team_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch teams data: {response.status_code}")
    teams_data = response.json()
    teams = teams_data.get("sports", [])[0].get("leagues", [])[0].get("teams", [])
    team_ids = [team["team"]["id"] for team in teams]

    seen_matches: Set[str] = set()  # To track already processed match IDs
    unique_matches = []

    # Fetch match details for each team and year
    for team_id in team_ids:
        url = team_schedule_url_template.format(TEAM_ID=team_id, YEAR=year)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch schedule for team {team_id} in year {year}: {response.status_code}")
            continue
        schedule_data = response.json()
        events = schedule_data.get("events", [])
        for event in events:
            match_id = event.get("id")
            if match_id in seen_matches:
                continue  # Skip duplicate matches
            seen_matches.add(match_id)

            competitions = event.get("competitions", [])
            if competitions:
                competition = competitions[0]
                competitors = competition.get("competitors", [])
                if len(competitors) == 2:
                    home_team = competitors[0]
                    away_team = competitors[1]
                    home_win = home_team.get("winner", False)
                    home_team_id = home_team.get("id")
                    away_team_id = away_team.get("id")
                    unique_matches.append(
                        (match_id, 1 if home_win else 0, home_team_id, away_team_id)
                    )

    return unique_matches


# print(getIDs([2023]))

# Get the play-by-play, given a game ID
def getPlayByPlay(game_id: str) -> List[Dict[str, Any]]:
    """
    Fetches play-by-play data for a given NFL game ID.

    Args:
        game_id (str): The unique ID of the game.

    Returns:
        List[Dict[str, Any]]: play information
    """
    play_by_play_url = f"https://cdn.espn.com/core/nfl/playbyplay?xhr=1&gameId={game_id}"

    # Fetch the play-by-play data
    response = requests.get(play_by_play_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch play-by-play data for game {game_id}: {response.status_code}")

    data = response.json()

    # Navigate to the play-by-play content
    drives = data.get("gamepackageJSON", {}).get("drives", {}).get("previous", [])
    if not drives:
        raise Exception(f"No play-by-play data found for game {game_id}")

    extracted_plays = []

    # Extract information from each drive
    for drive in drives:
        drive_data = drive.get("plays", [])
        extracted_plays.extend(drive_data)

    return extracted_plays

# Save everything in CSV
def save_game(game_id: str, data: List[Dict[str, Any]], directory: str) -> None:
    """
    Saves the play-by-play data for a given NFL game ID to a CSV file in the specified directory.

    Args:
        game_id (str): The unique ID of the game.
        data (List[Dict[str, Any]]): The play-by-play data to save.
        directory (str): The directory where the CSV file will be saved.
    """
    # Ensure the directory exists, create it if it doesn't
    os.makedirs(directory, exist_ok=True)

    # Normalize the data and create a DataFrame
    df = pd.json_normalize(data)
    
    # Construct the file path
    file_path = os.path.join(directory, f"game_{game_id}.csv")
    
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    
    print(f"CSV file for game {game_id} saved at {file_path}")

def getHomeWinProbabilities(game_id: str) -> List[Dict[str, Any]]:
    """
    Fetches home win probabilities for each sequenceNumber for a given game ID.

    Args:
        game_id (str): The unique ID of the game.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing sequenceNumber and homeWinProbability.
    """
    url = f"http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{game_id}/competitions/{game_id}/probabilities?limit=999"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch home win probabilities for game {game_id}: {response.status_code}")
    
    data = response.json()
    probabilities = data.get("items", [])
    
    extracted_probabilities = [
        {
            "sequenceNumber": prob.get("sequenceNumber"),
            "homeWinProbability": prob.get("homeWinPercentage")
        }
        for prob in probabilities
    ]
    
    return extracted_probabilities


if __name__ == "__main__":
    years = [2024]
    for year in years:
        matches = getIDs(year)
        print(f"Found {len(matches)} unique matches for year {year}")
        failures = 0
        for match in matches:
            match_id, home_win, home_team_id, away_team_id  = match
            try:
                play_by_play_data = getPlayByPlay(match_id)
                probabilities = getHomeWinProbabilities(match_id)
                merged_array = [{**d1, **d2} for d1, d2 in zip(play_by_play_data, probabilities[1:])]
                save_game(match_id, [{"home_team_id": home_team_id, "away_team_id": away_team_id, "home_win": home_win} | probabilities[0]] + merged_array, f"data/{year}")
            except Exception as e:
                print(str(e))
                failures += 1
        print(f"Failed to fetch play-by-play data for {failures} matches")
