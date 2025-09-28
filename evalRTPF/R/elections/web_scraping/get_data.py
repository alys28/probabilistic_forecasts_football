import requests

URL = "https://clob.polymarket.com/"
response = requests.get(url, params=querystring)

print(response.json())

# Hit the prices-history of Polymarket
def get_probabilities(market, startTs, endTs):
    """
    Args (defined by Polymarket API)
    - market (ID)
    - startTs
    - endTs
    """
    url = URL + "prices-history"
    response = requests.get(url, params = {"market": market, "startTs": startTs, "endTs": endTs})