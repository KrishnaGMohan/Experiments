import requests

from os import listdir
from os.path import isfile, join

# Replace <Subscription Key> with your valid subscription key.
subscription_key = "39aa3d13dc0345b9b8d948c81c92a5f1"
assert subscription_key

# You must use the same region in your REST call as you used to get your
# subscription keys. For example, if you got your subscription keys from
# westus, replace "westcentralus" in the URI below with "westus".
#
# Free trial subscription keys are generated in the westcentralus region.
# If you use a free trial subscription key, you shouldn't need to change
# this region.
vision_base_url = "https://eastus.api.cognitive.microsoft.com/vision/v1.0/"
ocr_url = vision_base_url + "ocr"

image_dir = "C:/temp/OCR"
files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

for file in files:
    headers    = {'Ocp-Apim-Subscription-Key': subscription_key,
                  'Content-Type': 'application/octet-stream'}
    params  = {'language': 'unk', 'detectOrientation': 'true'}
    image_data = open(join(image_dir, file), "rb").read()
    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)

    response.raise_for_status()
    analysis = response.json()

    # Extract the word bounding boxes and text.
    line_infos = [region["lines"] for region in analysis["regions"]]
    word_infos = []
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                word_infos.append(word_info)
    # print(word_infos)

    print(file, ",", (0,0) , ",", "BOIM")
    for word in word_infos:
        bbox = [int(num) for num in word["boundingBox"].split(",")]
        text = word["text"]
        origin = (bbox[0], bbox[1])
        print(file, ",", origin, ",", text)
    print(file, ",", (0,0) , ",", "EOIM")