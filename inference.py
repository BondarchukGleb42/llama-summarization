import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id =  "SalmanFaroz/Llama-2-7b-QLoRa-samsum"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def predict(dscr, pos_comms, neg_comms):
    """
    Функция для суммаризации информации о товаре и поиску его плюсов и минусов.
    Принимает на вход:
        -dscr: str, текстовое описание товара;
        -pos_comms: list, список позитивных комментариев;
        -neg_comms: list, список негативных комментариев.
    Генерирует промпт для модели, получает ответ и парсит его. Возвращает dict вида:
    {
        'pluses': list, список позитивных характеристик товара;
        'minuses': list, список негативных характеристик товара.
    }
    """
    prompt = "\n".join(["Найди плюсы и минусы товара по его текстовому описанию и отзывам.",
                      "### Товар:",
                      dscr,
                      "\n"
                      "Позитивные отзывы: ",
                      "\n".join([f"{i+1}) {txt}" for i, txt in enumerate(pos_comms)]),
                      "\n"
                      "Негативные отзывы: ",
                      "".join([f"{i+1}) {txt}" for i, txt in enumerate(neg_comms)]),
                      "### Плюсы и минусы:"])
    
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=400,
        )[0],
        skip_special_tokens=True
    )
    output = output[output.index("### Плюсы и минусы:")+19:]
    output = output[:output.index("###")+3]
    output = output.split("\n\n")
    
    pluses = output[0].split("\n")[2:]
    pluses = list(map(lambda x: x[x.index(")")+2:], pluses))
    minuses = output[1].split("\n")[1:-1]
    minuses = list(map(lambda x: x[x.index(")")+2:], minuses))
    
    return {"pluses": pluses, "minuses": minuses}