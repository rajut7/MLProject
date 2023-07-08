import requests
import json

url= "https://incomemodel-ed9c42c0b646.herokuapp.com/predict"

# explicit the sample to perform inference on
data =  { 'age':50,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Doctorate",
            'education_num':16,
            'marital_status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_weekk':50,
            'native_country':"United-States"
            }


# post to API and collect response
response = requests.post(url, json=data )

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())
