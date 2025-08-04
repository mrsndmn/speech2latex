import pandas as pd
import os
import time
import tqdm
import logging
from multiprocessing import Pool
from functools import partial
from ApiClient import APIClient

logging.basicConfig(
    filename='fast-main.log',  
    level=logging.INFO,         
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_excel_to_df(file_path,sheet_name = 0):
    ending = file_path.split(".")[-1]
    if ending == "csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path,sheet_name)

def try_get_audio(client, formula, contentType, person, output_file, _iter=0):
    if _iter == 20:
        log_msg = f" formula '{formula}' was skipped "
        logging.warning(log_msg)
        print(log_msg)
        return None

    try:        
        log_msg = f"creating {output_file}"
        logging.info(log_msg)
        print(log_msg)
        
        audio_data = client.request_data(formula, contentType, person)
        return audio_data

    except Exception as ex:
        log_msg = f"Error while getting formula '{formula}': {ex}"
        logging.error(log_msg)
        print(log_msg)
        client.get_access_token()
        return try_get_audio(client, formula, contentType, person, output_file, _iter=_iter+1)

        

def process_row_by_all_speakers(row, client, result_path, contentType, ru_speakers, person_index):
    id = row["FILENAME"]
    formula = row["INPUT:text"]

    if person_index == len(ru_speakers):
        person_index = 0

    output_file = f"./{result_path}/audio_{id}.wav"
    audio_data = try_get_audio(client, formula, contentType, ru_speakers[person_index], output_file)

    if audio_data:
        with open(output_file, 'wb') as file:
            file.write(audio_data)
        log_msg = f"File save {output_file}"
        logging.info(log_msg)
        print(log_msg)

    return person_index + 1




def process_row_by_one_speaker(row, client, result_path, contentType, person,cond_lang):
    id = row["id"]
    formula = row["pronunciation"]
    language = row["language"]

    output_leaf = os.path.join(result_path, person)
    os.makedirs(output_leaf,exist_ok=True)
    output_file = os.path.join(output_leaf, f"audio_{id}.wav")

    if os.path.exists(output_file):
        log_msg = f"Formula '{formula}' skipped/: {output_file}"
        logging.info(log_msg)
        print(log_msg)
        return 
    
    
    audio_data = try_get_audio(client, formula, contentType, person, output_file)
    if audio_data:
        with open(output_file, 'wb') as file:
            file.write(audio_data)
        log_msg = f"File {id} {formula} save {output_file}"
        logging.info(log_msg)
        print(log_msg)



def csv_worker(data_path, result_path, client_secret, token_url_access, verify_path, token_url_syntheze, cond_lang, speakers,sheet_name = 0):
    df = read_excel_to_df(data_path,sheet_name)

    client = APIClient(client_secret, verify_path, token_url_access, token_url_syntheze)
    client.get_access_token()
    contentType = "application/text"

    start_time = time.time()

    for person in speakers:
        process_row_partial = partial(process_row_by_one_speaker, client=client, result_path=result_path, contentType=contentType, person = person,cond_lang=cond_lang)

        with Pool(4) as pool:
            pbar = tqdm.tqdm(total=len(df))

            for _ in pool.imap_unordered(process_row_partial, [row for _, row in df.iterrows()]):
                pbar.update(1)

            pbar.close()

        elapsed_time = time.time() - start_time
        logging.info(f"duration {elapsed_time}s ")
        print(f"duration {elapsed_time} ")


if __name__ == "__main__":
    data_path = "../ASRDataCreator/ExtraASR"
    sheet_name = 0
    result_path = "path_output"

    os.makedirs(result_path,exist_ok=True)

    client_secret = "your data"
    token_url_access = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    verify_path = "./russian_trusted_root_ca_pem.crt" 
    token_url_syntheze = "https://smartspeech.sber.ru/rest/v1/text:synthesize"

    # ru_speakers = [
    #     "Nec_24000","Bys_24000","May_24000","Tur_24000",
    #                "Ost_24000","Pon_24000"] 
    eng_speakers = ["Kin_24000"]
    cond_lang = "eng"


    csv_worker( data_path,result_path,client_secret,token_url_access,verify_path,token_url_syntheze,cond_lang = cond_lang,speakers=eng_speakers,sheet_name = sheet_name)
