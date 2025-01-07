'''
Two links:
    https://cdn.espn.com/core/nfl/schedule?xhr=1&year={YEAR}&week={WEEK_NO} where 1 <= WEEK_NO <= 18 -- Get all the games with their IDs
    https://cdn.espn.com/core/nfl/playbyplay?xhr=1&gameId=401671830 where we get all the play-by-play information
'''
import requests


# Get all the games with their ids
def getIDs(years):
    '''
        years: List[int]
    '''
    for year in years:
        for week_no in range(1, 19):
            url = f"https://cdn.espn.com/core/nfl/schedule?xhr=1&year={year}&week={week_no}" 
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Request for week {week_no}, year {year} was successful!")
                    data = response.json()
                    schedule = data["content"]["schedule"]
                    for date in schedule:
                        for game in date["games"]:
                            if game["competitions"]["playByPlayAvailable"]:
                                id = game["competitions"]["id"]

                else:
                    print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
            except:
                print("Failed to get IDs")
