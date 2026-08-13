https://www.usbr.gov/uc/water/hydrodata/reservoir_data/1998/csv/25.csv
https://www.usbr.gov/uc/water/hydrodata/reservoir_data/913/csv/25.csv


#####################
# get reservoir evap
#####################

import requests

url = "https://operevap.dri.edu/auth/request_key"
params = {
    "name": "Will Keenan",
    "email": "william.keenan@colostate.edu",
    "justification": "Research on naturalized streamflow in the Upper Colorado River Basin"
}

response = requests.get(url, params=params)
print(response.text)