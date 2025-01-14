import requests
from typing import List, Tuple, Set

def getIDs(years: List[int]) -> List[Tuple[str, int, str, str]]:
    """
    Fetches unique match data for NFL teams for the given years, including match ID, home win status, and team IDs.

    Args:
        years (List[int]): A list of years for which to fetch the match data.

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
        for year in years:
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


print(getIDs([2023]))

# Get the play-by-play, given a game ID
def getPlayByPlay(id: str) -> List[str]:
    pass


# Save everything in CSV