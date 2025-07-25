import requests
import uuid
from datetime import datetime
import os
class APIClient:
    def __init__(self,client_secret,verify_path,token_url_access,token_url_syntheze):
        self.client_secret = client_secret
        self.verify_path  = verify_path
        self.token_url_access = token_url_access
        self.token_url_syntheze = token_url_syntheze
        self.access_token = None
        self.expires_at = None

    def get_access_token(self):

        if not os.path.exists(self.verify_path):
            print("No certificate")
            return 
    
        rq_uid = str(uuid.uuid4())

        
        headers = {
            'Authorization': "Basic " + self.client_secret,
            'RqUID': rq_uid,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        print("header",headers)

        
        payload = {
            'scope': 'SALUTE_SPEECH_PERS',
        }

        
        response = requests.post(self.token_url_access, headers=headers, data=payload,verify=self.verify_path)

        if response.status_code == 200:
            response_data = response.json()
            self.access_token = response_data.get('access_token')
            expires_at_unix = response_data.get('expires_at')

            self.expires_at = datetime.fromtimestamp(expires_at_unix / 1000)
            
            print(f"Access token until {self.expires_at}")
        else:
            print("Error getting access token:", response.json())


    def request_data(self,text,contentType = "application/text",speaker= "Nec_24000"):

        if not self.access_token or (self.expires_at and datetime.now() > self.expires_at):
            print("No access token.")
            print(f"now {datetime.now()} expires_at {self.expires_at}")
            self.get_access_token()

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': contentType
        }

        params = {
            "voice":speaker,

        }
        
        response = requests.post(self.token_url_syntheze, headers=headers,params=params,data=text.encode('utf-8'),verify=self.verify_path)

        if response.status_code == 200:
            x_request_id = response.headers.get('X-Request-ID')
            content_type = response.headers.get('Content-Type')

            print(f"Got responce. X-Request-ID: {x_request_id}, Content-Type: {content_type}")

            audio_data = response.content

            return audio_data

        else:
            print("Some error happened:")
            print(response.json())



if __name__ == "__main__":
    client_secret = "your Authorization data"
    token_url_access = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    verify_path = "./russian_trusted_root_ca_pem.crt" 

    token_url_syntheze = "https://smartspeech.sber.ru/rest/v1/text:synthesize" 
    text = "test X + Y = 12312312312"

    client = APIClient(client_secret, verify_path, token_url_access, token_url_syntheze)
    client.get_access_token()
    audio_data = client.request_data(text)
