import requests

BOT_TOKEN  = '1351157852:AAGkg3vEHDKZ60ACGxUesaagbIU3CDVgkIA' # Insert your bot token here 
BOT_CHATID = '461197140' # This is the chat between your account and the bot

def sendtext(bot_message):
    send_text = 'https://api.telegram.org/bot' + BOT_TOKEN + '/sendMessage?chat_id=' + BOT_CHATID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)

    return response.json()

def sendImage(img):

    send_text = 'https://api.telegram.org/bot' + BOT_TOKEN + '/sendPhoto?chat_id=' + BOT_CHATID + '&photo='

    data={'photo': img.read()}
    response = requests.post(send_text, files=data)

    return response.json()