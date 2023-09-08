import telebot

from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer
)
from peft import PeftModel

bot = telebot.TeleBot('6698787409:AAGHismbsqP6fi7ibbp1n9uUxA9FafBw5-w')

tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = PeftModel.from_pretrained(model, 'ivankadchenko/my_tg_bot')
model.eval()


def generate(prompt):
    data = tokenizer(prompt, return_tensors='pt')
    output_ids = model.generate(
        **data,
        top_k=50,
        top_p=0.95,
        num_beams=3,
        do_sample=True,
        no_repeat_ngram_size=16,
        temperature=1.2,
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=50257,
        max_new_tokens=40,
        min_new_tokens=1
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    output = output[:output.find('@@')]
    return output

@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}! Напиши что-нибудь, чтобы начать наш диалог.')

@bot.message_handler(content_types=['text'])
def model_response(message):
    ans = generate('@@ПЕРВЫЙ@@ ' + message.text.lower() + ' @@ВТОРОЙ@@')
    bot.send_message(message.chat.id, ans)


bot.polling(none_stop=True)